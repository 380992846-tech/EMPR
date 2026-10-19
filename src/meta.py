"""运行元数据（复现性）。

每个 ``results/*.json`` 都要能回答"这份数字是怎么来的"：哪个 commit、什么环境、
哪套提示词、多少题、什么解码参数。缺了这些，云上跑完回过头对不上号——尤其是
提示词改过之后。所有 run_* 脚本统一用这里的 ``result_meta()`` 落盘。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    """短 commit + 是否有未提交改动（脏工作区跑出来的结果必须能看出来）。"""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        tag = sha.stdout.strip() or "unknown"
        return tag + ("+dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(text: str) -> str:
    """模板/配置指纹：改没改过提示词，一眼可查。"""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]


def model_slug(name: str) -> str:
    """把模型标识转成安全的文件名片段。

    同时支持两种传入方式：
      * HF id：``Qwen/Qwen2.5-7B-Instruct``          -> ``Qwen__Qwen2.5-7B-Instruct``
      * 本地权重目录：``/root/models/Qwen2.5-7B-Instruct`` -> ``models__Qwen2.5-7B-Instruct``
    只用末两段，避免绝对路径把文件名搞成一长串。完整标识仍写在 json 的 ``model`` 字段里。
    """
    parts = [p for p in str(name).replace("\\", "/").strip("/").split("/") if p]
    if not parts:
        return "unknown"
    return "__".join(parts[-2:]) if len(parts) >= 2 else parts[0]


def env_summary() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        info["torch"] = None
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception:
        info["vllm"] = None
    return info


def result_meta(script: str, **extra) -> dict:
    """拼装一份可复现的元数据块；extra 里放模型/基准/解码参数等。"""
    meta = {"script": script, "created_utc": utc_now(), "git_commit": git_commit(),
            "env": env_summary(), "argv": sys.argv[1:]}
    meta.update({k: v for k, v in extra.items() if v is not None})
    return meta


def _sanitize(obj):
    """把非有限浮点（NaN/±inf）换成 None。

    统计量在退化情形下会产生 NaN（例如 McNemar 两组完全一致、没有配对差异），
    而 ``json.dumps`` 默认会把它写成裸 ``NaN`` —— 那不是合法 JSON（RFC 8259），
    R / jq / 严格解析器都会读失败。结果文件必须能被任何工具读。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def write_json(path, obj) -> Path:
    """统一的落盘方式（utf-8 + 缩进 + 中文不转义 + 非有限值转 null）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_sanitize(obj), ensure_ascii=False, indent=2, allow_nan=False)
    p.write_text(text, encoding="utf-8")
    return p
