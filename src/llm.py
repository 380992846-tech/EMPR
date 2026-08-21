"""LLM 推理封装：vllm（快）优先，transformers 兜底。

统一接口：
    gen_batch(prompts: list[str]) -> list[str]
"""
from __future__ import annotations


def make_generator(model_name, backend="auto", gpus=1, seed=42, max_new_tokens=1024,
                   dtype="auto", device_map="auto"):
    """返回一个带 .gen_batch(prompts) 的对象。"""
    if backend in ("auto", "vllm"):
        try:
            return _VLLMGenerator(model_name, gpus=gpus, seed=seed, max_tokens=max_new_tokens)
        except Exception as e:
            print(f"[llm] vllm 不可用({e})，退回 transformers")
            if backend == "vllm":
                raise
    return _HFGenerator(model_name, seed=seed, max_new_tokens=max_new_tokens,
                        dtype=dtype, device_map=device_map)


# ---------------------------------------------------------------- vllm
class _VLLMGenerator:
    def __init__(self, model_name, gpus=1, seed=42, max_tokens=1024):
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=model_name, tensor_parallel_size=gpus,
                        seed=seed, dtype="auto")
        self._params = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                      seed=seed)

    def gen_batch(self, prompts, batch_size=128):
        from vllm import SamplingParams
        out = []
        for i in range(0, len(prompts), batch_size):
            res = self._llm.generate(prompts[i:i + batch_size], self._params)
            out.extend(r.outputs[0].text for r in res)
        return out


# ---------------------------------------------------------------- transformers
class _HFGenerator:
    def __init__(self, model_name, seed=42, max_new_tokens=1024, dtype="auto", device_map="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        torch.manual_seed(seed)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def gen_batch(self, prompts, batch_size=8):
        import torch
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=4096).to(self._model.device)
            with torch.no_grad():
                gen = self._model.generate(**enc, do_sample=False,
                                           max_new_tokens=self._max_new_tokens,
                                           pad_token_id=self._tok.pad_token_id)
            texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
            out.extend(texts)
        return out
