"""因果追踪的**集成**测试：在微型随机模型上真跑一遍 hook/恢复/对齐。

为什么需要它：单元测试只能验证纯函数，而 Week 3 最容易出错的地方恰恰是张量形状与
hook 语义——比如解码步（h.shape[1] == 1）被误当成 prefill 去覆盖 hidden、
或者 clean/corrupt 长度不同时对齐错位。这些只有真跑前向才暴露得出来。

用 ``hf-internal-testing/tiny-random-gpt2``（很小，通常已在 HF 缓存里）；
没网/没缓存时自动 skip，不会让 CI 变红。

注意：随机权重的模型当然"答不对题"，所以这里**不断言恢复率**，只断言
  * 每层都能采到 hidden；
  * 恢复只改目标层（其余层与 corrupt 一致）；
  * 对齐函数语义正确；
  * 恢复干预不会把形状搞坏（前向能跑完并产出文本）。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    except Exception as e:  # pragma: no cover - 无网/无缓存
        pytest.skip(f"无法加载 {TINY_MODEL}: {type(e).__name__}: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def test_layers_found(causal, tiny):
    model, _ = tiny
    layers = causal._layers(model)
    assert len(layers) >= 2


def test_align_right_semantics(causal):
    torch.manual_seed(0)
    c = torch.randn(1, 4, 3)
    # 等长：原样返回
    assert torch.equal(causal._align_right(c, 4), c)
    # clean 更长：截掉左端（保留末尾）
    out = causal._align_right(c, 2)
    assert out.shape == (1, 2, 3)
    assert torch.equal(out, c[:, 2:, :])
    # clean 更短：左侧补零（末尾对齐）
    out = causal._align_right(c, 6)
    assert out.shape == (1, 6, 3)
    assert torch.all(out[:, :2, :] == 0)
    assert torch.equal(out[:, 2:, :], c)


def test_hook_patches_only_target_layer(causal):
    """hook 的三条不变量（不需要模型，直接喂张量）：

      1. 目标层 + prefill -> 返回 clean（按右端对齐）；
      2. 非目标层 -> 原样返回（"只恢复目标层"的核心语义）；
      3. 解码步（shape[1] == 1）-> 原样返回（不能把 clean 的位置 0 写进新 token）。
    """
    torch.manual_seed(0)
    clean = {5: torch.randn(1, 9, 4)}
    prefill = torch.randn(1, 6, 4)     # 比 clean 短 -> 左补零右对齐

    out_target = causal.make_restore_hook(5, 5, clean)(None, (), prefill)
    assert out_target.shape == prefill.shape
    assert torch.allclose(out_target, causal._align_right(clean[5], 6), atol=1e-6)

    out_other = causal.make_restore_hook(4, 5, clean)(None, (), prefill)
    assert torch.equal(out_other, prefill), "非目标层必须原样保留 corrupt 状态"

    decode = torch.randn(1, 1, 4)
    out_decode = causal.make_restore_hook(5, 5, clean)(None, (), decode)
    assert torch.equal(out_decode, decode), "解码步绝不能被 clean hidden 覆盖"

    # 目标层不在 clean_states 里时也要安全返回
    out_missing = causal.make_restore_hook(5, 5, {})(None, (), prefill)
    assert torch.equal(out_missing, prefill)


def test_clean_states_and_restore_run(causal, tiny):
    """真跑一遍：每层都能采到 prefill hidden，带恢复的生成能跑完且形状正确。"""
    model, tok = tiny
    prompt = "Question: what is 2 + 2? Answer with ANSWER: line."
    n_layers = len(causal._layers(model))

    with torch.no_grad():
        states = causal.collect_clean_states(model, tok, prompt, "cpu")
    assert len(states) == n_layers, "每层都应采到 prefill hidden"
    seq_len = next(iter(states.values())).shape[1]
    assert seq_len > 1 and all(s.shape[1] == seq_len for s in states.values())

    text = causal.run_once(model, tok, prompt, "cpu", max_new=4)
    assert isinstance(text, str)

    target = n_layers // 2
    text_r = causal.run_restore(model, tok, prompt, target, states, "cpu", max_new=4)
    assert isinstance(text_r, str)

    # corrupt 输入（含占位符）也要能安全跑完：不同 token 长度触发 _align_right
    corrupt_prompt = prompt.replace("2 + 2", "zork blork")
    text_c = causal.run_restore(model, tok, corrupt_prompt, target, states, "cpu", max_new=4)
    assert isinstance(text_c, str)


def test_decode_step_is_not_patched(causal, tiny):
    """解码步（序列长度 1）确实会出现，且不会被 hook 改写（见上面的不变量 3）。"""
    model, tok = tiny
    prompt = "2+2?"
    with torch.no_grad():
        states = causal.collect_clean_states(model, tok, prompt, "cpu")
    seen_lens = []

    def hook(module, args, output):
        seen_lens.append(causal._unpack_hidden(output).shape[1])

    layers = causal._layers(model)
    handles = [layers[0].register_forward_hook(hook)]
    try:
        with torch.no_grad():
            causal.run_restore(model, tok, prompt, 0, states, "cpu", max_new=3)
    finally:
        for h in handles:
            h.remove()
    assert 1 in seen_lens, "解码步应当出现长度为 1 的前向"
    assert max(seen_lens) > 1, "prefill 步应当出现长序列前向"
