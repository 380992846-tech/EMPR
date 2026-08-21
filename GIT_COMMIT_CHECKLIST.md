# 提交 Git 前最终检查清单

提交前逐条勾，避免把脏东西/坏代码/假数据推进仓库。
**原则：仓库里只放"代码 + 文档 + 真实小结果"，不放大模型权重、不放大数据、不放大崩溃日志、不含任何未核实的编造数字。**

---

## 0. 内容与诚信（最重要，先想清楚）
- [ ] **所有 `results/*.json` 都是真实跑出来的**，不含手填/伪造数字。
- [ ] 没有把 AI 稿里的旧空数字/占位数字混进来当结果。
- [ ] 没有把模型权重、HF 缓存、日志大文件塞进仓库。
- [ ] README / 论文草稿里没有未披露的"模拟数据"冒充实证的地方。

## 1. 版本 / 环境
- [ ] `.gitignore` 已覆盖：`__pycache__/`、`.venv/`、`models/`、`data/*.jsonl`、`results/*.json`（保留 `.gitkeep`）。
- [ ] `requirements.txt` 与代码实际用的一致（新增 import 都已补进）。
- [ ] 关键依赖 `sympy/numpy/scipy/datasets/vllm` 都在 requirements（可选的在注释里说明）。

## 2. 代码质量
- [ ] 所有 `.py` 能编译：`python -m py_compile src/*.py scripts/*.py`
- [ ] 判定自测通过：`python -m src.grading`（7/7 OK）。
- [ ] 没有残留 `print` 调试、`TODO`、`pass  # placeholder`（有就删或写清）。
- [ ] 没有绝对路径硬编码（如 `C:\Users\...`）——全用相对 `ROOT / ...`。
- [ ] 没有把本机个人路径/邮箱/密钥写进代码或 config。

## 3. 数据
- [ ] `data/` 目录**不进库**（太大且可再生）——由 `download_data.py` 生成。
- [ ] README 写清"如何重新下载数据"（命令已写）。
- [ ] 若放了少量样例数据，标清来源与许可（SciBench/OlympiadBench 需注明出处与授权）。

## 4. 结果
- [ ] `results/` 只提交**小的、真实的**东西（如 `table.md`、汇总 json）；大 raw 输出用 `.gitignore` 排除或放仓库外。
- [ ] 每个提交带说明：这次跑了哪个模型/基准/多少题、CI 多少。

## 5. 文档
- [ ] README 的"快速开始"命令与当前脚本名一致（改名过要同步）。
- [ ] DEPLOY_CHECKLIST.md 没有过期的命令/路径。
- [ ] 若有 `config.yaml` 改动，README 相应说明同步。

## 6. 提交前快速自查命令
```bash
# 编译检查
python -m py_compile src/*.py scripts/*.py
# 判定自测
python -m src.grading
# 看 git 会提交什么（确认没有大文件/缓存）
git status
git add -A
git diff --cached --stat      # 应只看到代码 + 文档 + 小结果
# 若混进大文件，用 .gitignore 排除后再 add
```

## 7. 提交信息规范
- 每条提交用一句描述做了什么 + 结果来源，例如：
  - `feat(eval): add Week1 run_eval pipeline + grading self-test`
  - `data: SciBench physics 323 problems via GitHub downloader`
  - `results: Qwen-14B scibench cot n=200 acc=0.52 [0.45,0.59]`
- 不要提交"final_final_v2"这种无意义版本。

---

> 目标：**任何拿到你仓库的人，clone 下来跑 README 的命令就能复现你的结果。**
> 这条做不到，论文的"可复现性"就不成立——这也是审稿人最看重的。
