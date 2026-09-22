"""LLM 推理封装：vllm（快，云上首选）/ transformers（兜底，CPU亦可用）。

统一接口::

    gen = make_generator(model_name, backend="auto", ...)
    outs = gen.gen_batch(prompts)      # list[str] -> list[str]，顺序与长度严格一致

三条容易静默出错的约定（都踩过）：
  1. **顺序与长度严格一致**：任何一条失败也要占位（返回 ``""``），否则
     ``zip(rows, outputs)`` 会整体错位，把 A 题的输出判到 B 题的答案上。
     （现在 gen_batch 之后还会校验长度，不一致直接抛错。）
  2. **批量生成必须左填充**：decoder-only 模型右填充时会在 pad 之后续写，
     transformers 只打一行 warning 不报错，批量结果直接是垃圾。
  3. **instruct 模型要走 chat template**：直接把裸 prompt 喂给 instruct 模型，
     它会去"续写指令"而不是"回答问题"（实测 0.5B 会回 ``ANSWER: ANSWER: 4.``）。
     默认 auto：tokenizer 自带 chat_template 就用，``chat=False`` 可关。

transformers 版本兼容：``from_pretrained`` 的精度参数在 v4 叫 ``torch_dtype``、v5 改成
``dtype``，且两个版本都是 ``**kwargs``，无法用签名探测。这里按主版本号选择，避免
"参数被当成 config 属性静默忽略 -> 模型以 fp32 加载 -> 云上 14B 直接 OOM"。
"""
from __future__ import annotations

import os
import time


def _dtype_kwarg() -> str:
    try:
        import transformers
        major = int(str(transformers.__version__).split(".")[0])
    except Exception:  # pragma: no cover
        return "torch_dtype"
    return "dtype" if major >= 5 else "torch_dtype"


def resolve_device(device: str = "auto") -> str:
    """'auto' -> 有 CUDA 用 cuda，否则 cpu。"""
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _as_chat(tokenizer, prompts):
    """把裸 prompt 列表转成 chat 格式字符串（走 tokenizer 自带的 chat template）。"""
    out = []
    for p in prompts:
        try:
            out.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True))
        except Exception:
            out.append(p)
    return out


def make_generator(model_name, backend="auto", gpus=1, seed=42, max_new_tokens=1024,
                   dtype="auto", device_map=None, device="auto", batch_size=8,
                   chat="auto", verbose=True):
    """返回一个带 .gen_batch(prompts) 的对象。chat: auto|True|False。"""
    if backend in ("auto", "vllm"):
        try:
            return _VLLMGenerator(model_name, gpus=gpus, seed=seed,
                                  max_tokens=max_new_tokens, chat=chat, verbose=verbose)
        except ModuleNotFoundError:
            if backend == "vllm":
                raise
            if verbose:
                print("[llm] vllm 未安装，退回 transformers")
        except Exception as e:
            if backend == "vllm":
                raise
            if verbose:
                print(f"[llm] vllm 初始化失败({type(e).__name__}: {e})，退回 transformers")
    return _HFGenerator(model_name, seed=seed, max_new_tokens=max_new_tokens,
                        dtype=dtype, device_map=device_map, device=device,
                        batch_size=batch_size, chat=chat, verbose=verbose)


# ---------------------------------------------------------------- vllm
class _VLLMGenerator:
    def __init__(self, model_name, gpus=1, seed=42, max_tokens=1024, chat="auto",
                 gpu_memory_utilization=0.90, verbose=True):
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=model_name, tensor_parallel_size=gpus, seed=seed,
                        dtype="auto", gpu_memory_utilization=gpu_memory_utilization,
                        trust_remote_code=True)
        self._params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
        self.name = model_name
        self.backend = "vllm"
        tok = self._llm.get_tokenizer()
        self.chat = bool(getattr(tok, "chat_template", None)) if chat == "auto" else bool(chat)
        self._tok = tok
        if verbose:
            print(f"[llm] vllm 已加载 {model_name}（chat_template={self.chat}）")

    def gen_batch(self, prompts, batch_size=128):
        if self.chat:
            prompts = _as_chat(self._tok, prompts)
        out = []
        for i in range(0, len(prompts), batch_size):
            res = self._llm.generate(prompts[i:i + batch_size], self._params)
            for r in res:
                if not getattr(r, "outputs", None):
                    # vllm 对请求级错误会返回空 outputs，兜底占位避免整批崩溃/错位
                    out.append("")
                    continue
                out.append(r.outputs[0].text)
        return out


# ---------------------------------------------------------------- transformers
class _HFGenerator:
    def __init__(self, model_name, seed=42, max_new_tokens=1024, dtype="auto",
                 device_map=None, device="auto", batch_size=8, chat="auto", verbose=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.name = model_name
        self.backend = f"transformers/{self.device}"
        # CPU 上把线程用满（否则 transformers 默认单线程，慢到不可用）。
        # 但不要超过 EMPR_TORCH_THREADS（默认 8）：本机同时跑两个任务时，
        # 两边各开 16 线程会互相抢占，实测比单任务慢一个数量级。
        if self.device == "cpu":
            cap = int(os.environ.get("EMPR_TORCH_THREADS", "8"))
            torch.set_num_threads(max(1, min(cap, os.cpu_count() or 4)))

        self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # **必须左填充**：decoder-only 模型批量生成时，右填充会让模型在 pad 之后继续
        # 续写，直接产出错位/垃圾结果（transformers 只会打一行 warning，不报错）。
        # 单题跑看不出来，只有 batch>1 才中招——早期版本这里就是右填充。
        self._tok.padding_side = "left"
        self.chat = bool(getattr(self._tok, "chat_template", None)) if chat == "auto" else bool(chat)

        kwargs = {_dtype_kwarg(): dtype}
        # device_map 需要 accelerate；没装就直接 .to(device)
        if device_map:
            kwargs["device_map"] = device_map
        torch.manual_seed(seed)
        if verbose:
            print(f"[llm] 加载 {model_name}（{self.backend}，{_dtype_kwarg()}={dtype}，"
                  f"batch={batch_size}，chat_template={self.chat}）...")
        self._model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not device_map:
            self._model = self._model.to(self.device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def gen_batch(self, prompts, batch_size=None):
        import torch
        if self.chat:
            prompts = _as_chat(self._tok, prompts)
        bs = batch_size or self.batch_size
        # CPU 上 8 条一批反而更慢（padding 到最长），默认降到 4
        if self.device == "cpu":
            bs = min(bs, 4)
        out = []
        n = len(prompts)
        t0 = time.time()
        for i in range(0, n, bs):
            batch = prompts[i:i + bs]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=4096).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, do_sample=False,
                                           max_new_tokens=self._max_new_tokens,
                                           pad_token_id=self._tok.pad_token_id)
            out.extend(self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True))
            done = min(i + bs, n)
            if n > bs:
                rate = (time.time() - t0) / max(done, 1)
                print(f"[llm] {done}/{n} 条 ({(time.time() - t0):.0f}s, "
                      f"预计剩余 {rate * (n - done):.0f}s)", flush=True)
        return out
