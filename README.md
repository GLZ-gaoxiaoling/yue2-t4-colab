# YuE2 on Colab Tesla T4

在 **免费 Colab 的 Tesla T4（16GB, sm_75）** 上跑通 [YuE2-3B](https://github.com/multimodal-art-projection/YuE) 全套：歌词 + 风格 → 可编辑乐谱 → 48kHz 立体声歌曲。

官方要求是 `BF16-capable GPU + 24GB VRAM`，T4 三条都不满足，所以本仓库是**打补丁的实测方案**，不是官方支持路径。实测一首 56.8 秒的歌，全流程约 4.5 分钟，峰值显存 8.7GB。

> 试听实测产物：[sample/audio.flac](sample/audio.flac)（56.84s，D 大调 88 BPM，48kHz 立体声）

## 结论先行

| 门槛 | 官方要求 | Tesla T4 | 应对 |
|---|---|---|---|
| BF16 张量核心 | sm_80+ | sm_75 只有 FP16 张量核心 | 全局改 FP16 |
| FP8 量化（唯一官方省显存开关） | sm_89+ | 不支持 | 放弃，改用分阶段解码 |
| FlashAttention / cuDNN attention | sm_80+ | 不支持 | 强制回退 SDPA |
| 显存 | 官方建议 24GB | 16GB | 实测峰值 8.7GB，够用 |
| 系统内存 | — | 免费档 12.7GB | 分阶段解码，避免 OOM |

## 快速开始

### 1. 装 Colab CLI（Windows 用户走 WSL）

`google-colab-cli` **官方只支持 Linux / macOS**，Windows 上必须进 WSL：

```bash
uv tool install google-colab-cli
```

### 2. 认证

```bash
colab sessions
```

按提示在浏览器打开链接、同意授权，把授权码粘回终端。注意授权页上的权限项**全部保持勾选**（尤其 Colab 相关），否则拿到的 token 缺 scope。

### 3. 建 T4 会话并连接

```bash
colab new -s ttt --gpu T4      # 免费档同时只能有 1 个 assignment
colab ssh  -s ttt              # 交互 shell，默认落在 /content
```

> CLI 只能管理**自己创建**的会话。`colab sessions` 里看到别人的 assignment 时，`colab status -s <id>` 会报 `not found`，需要去网页端「断开并删除运行时」释放配额。

### 4. 上传脚本并执行

```bash
# 在 WSL 里执行（路径换成你的 clone 位置）
for f in 00_probe 01_setup 02_patch_fp16 03_patch_attention 04_generate_staged 05_analyze; do
  colab upload -s ttt ./scripts/$f.py /content/$f.py
done

colab exec -s ttt -f /content/00_probe.py   # 确认 GPU / 内存 / 磁盘
python /content/01_setup.py                 # 后台 clone + pip install，日志 /content/setup.log
python /content/02_patch_fp16.py            # BF16 -> FP16
python /content/03_patch_attention.py       # 禁用 FlashAttention / cuDNN
python /content/04_generate_staged.py       # 生成 + 分阶段解码，日志 /content/gen.log
python /content/05_analyze.py               # 客观体检（RMS / 峰值 / 频谱质心）
```

### 5. 拉回产物

```bash
colab download -s ttt /content/outputs/song1/audio.flac ./audio.flac
colab download -s ttt /content/outputs/song1/score.abc  ./score.abc
```

## 四处补丁，为什么必须打

### 1. `02_patch_fp16.py` — BF16 降级 FP16

`pipeline.py` 里是硬拦截，不是警告：

```python
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The unquantized preset requires CUDA BF16 support")
```

torch 2.10 的 `is_bf16_supported()` 默认 `including_emulation=True`，在 T4 上会返回 `True`，但那是**用 FP32 软件模拟**，慢且不保证数值行为。脚本同时把 `torch.bfloat16` 全量替换为 `torch.float16`，走真正的 FP16 张量核心。

> 唯一的 FP8 路径在 `quantization.py` 里写死 `compute capability >= 8.9`，T4 完全用不了，脚本刻意跳过该文件。

### 2. 卸载 `torchvision` / `torchaudio`（`01_setup.py` 已内置）

YuE2 钉死 `torch==2.10.0`，而 Colab 预装的 `torchvision 0.26` 是为 torch 2.11 编译的。降级 torch 后：

```
RuntimeError: operator torchvision::nms does not exist
ModuleNotFoundError: Could not import module 'PreTrainedModel'
```

因为 transformers 会 `import torchvision`，而它对 `ImportError` 有兜底、对 `RuntimeError` 没有。卸掉这两个包，YuE2 根本不需要它们。

### 3. `03_patch_attention.py` — 禁用 FlashAttention

`cuda_graph.py` 只检查算子是否存在，没检查算力等级：

```python
flash = fused and config.head_dim <= 256 and hasattr(torch.ops.aten, "_flash_attention_forward") and ...
```

在 T4 上会一路走到 `torch.ops.aten._flash_attention_forward`，然后报 `FlashAttention only supports Ampere GPUs or newer`。补丁加上 `compute capability >= 8` 判断，并去掉 cuDNN 分支（T4 同样不支持），回退到 SDPA。

### 4. `04_generate_staged.py` — 分阶段解码

一条龙跑到底会在这里死掉：

```
[YuE2] Starting Loading audio decoder: elapsed 0.0s
Killed   (Exit 137)
```

`decode()` 会先把 MoT 挪到 CPU，再加载 VAE 解码器 —— 12.7GB 内存里同时躺着 7.2GB MoT + 解码器，被 OOM Killer 干掉。脚本改成：

```
plan → generate_semantic → synthesize → 存 latent.npy
                                     ↓
                    释放 MoT (pipe._model = None) + gc + empty_cache
                                     ↓
                           decode(latent) → audio.flac
```

顺带的好处：**latent 已经落盘，改解码参数不用重新生成**（想换 `vae_core_frames` 或换 decoder 时直接重跑解码段即可）。

## 实测数据

T4（16GB，免费档，FP16 + SDPA，`memory_budget_gib=12`, `vae_core_frames=512`）：

| 阶段 | 数值 |
|---|---|
| 模型加载 | 40s（权重已缓存） |
| 规划乐谱 | 425 tokens, ~14.6 tok/s |
| 语义 token | 1421 tokens, ~8.9 tok/s |
| 声学合成 | 32 步, 58.8s |
| VAE 解码 | 3 chunks, 5.4s |
| **总计** | **约 4.5 分钟 / 56.84s 音频** |
| 峰值显存 | 8.7GB / 15.3GB |
| 音频体检 | RMS -20.6 dBFS, 峰值 -2.19 dBFS, 零削波 |

对照：社区 PR [#174](https://github.com/multimodal-art-projection/YuE/pull/174) 在 RTX 4050（6GB, sm_89, BF16）上是 178.8s 出 56.7s 音频。T4 慢约 1.5 倍，主要输在没有 FlashAttention。

## 进阶：把任意歌曲改编成交响乐（cover 流程）

官方 cover 路线 = SheetSage2 转谱 → Vocal 旋律搬到 Ins → YuE2 重新配器。

```bash
# 1) 独立环境装 SheetSage2（Py3.11 + torch 2.8，与 YuE2 的 torch 2.10 隔离）
colab upload -s ttt ./scripts/10_setup_sheetsage.sh /content/setup_sheetsage.sh
nohup bash /content/setup_sheetsage.sh > /content/sheetsage_setup.log 2>&1 &

# 2) 原曲转 wav 后转谱（melody-only 会去掉和弦符号）
ffmpeg -y -i /content/input/source.mp3 -vn -ac 2 -ar 44100 /content/input/source.wav
cd /content/YuE
.venv-sheetsage2/bin/python models/SheetSage2/infer.py /content/input/source.wav \
  --output /content/cover/score --melody-only

# 3) 合并成纯音乐条件（Vocal 音符搬到 Ins），并用官方 abc_tools 校验
python /content/11_merge_instrumental.py /content/cover/score/score.abc \
  /content/cover/score/instrumental.abc
python /content/12_validate_abc.py

# 4) 旋律条件 + 交响风格生成
python /content/13_generate_cover.py

# 5) 体检（脚本接受任意音频路径）
python /content/05_analyze.py /content/outputs/symphony/audio.flac
```

三个必须知道的点：

- **长曲必须关 CFG**：`13_generate_cover.py` 里设了 `cfg_scale=1.0`。默认 CFG 是双分支，KV cache 翻倍，T4 上跑到 1400 tokens 就 `CUDA out of memory`；关掉后显存稳定在 8.8GB，2830 tokens 跑完不炸。
- **合并要算对八度记号**：`d'16` 是高八度十六分音符，`B,2` 是低八度二分音符。只按 `数字` 算时长会把小节算短，`abc_tools` 立刻报 `duration ... != meter duration`。
- **官方 `scripts/instrumental.py` 没随 v1.2.0 发布包提供**（SKILL.md 引用了但包里没有），所以本仓库用 `11_merge_instrumental.py` 复刻同样规则：Vocal 全音符搬到 Ins，Vocal 只留休止，同小节两轨都有音时 Vocal 优先。

实测（《天ノ弱》，原曲 110.4s）：

| 项 | 数值 |
|---|---|
| 转谱 | SheetSage2 melody-only，B♭ 小调 / 200 BPM / 96 小节 |
| 输出 | 113.2s，未截断 |
| 语义 token | 2830，401.5s |
| 声学合成 | 32 步，233.8s |
| **全流程** | **648s（约 10.8 分钟）** |
| 峰值显存 | 11.9GB |
| 音频体检 | RMS -15.4 dBFS，峰值 -0.14 dBFS，零削波 |

> 转谱误差会原样带进编曲；这类曲谱由 SheetSage2 提供，不是 YuE2 规划出来的。


## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `Allocation refused (precondition failed)` | 免费档只能 1 个 assignment，去网页端删掉正在跑的运行时 |
| `RuntimeError: operator torchvision::nms does not exist` | 没卸 torchvision |
| `FlashAttention only supports Ampere GPUs or newer` | 没打注意力补丁，或装了 `--backend vllm` |
| `Exit 137` / `Killed` | 系统内存 OOM，必须走分阶段解码 |
| 长曲跑到一半 `CUDA out of memory` | 默认 CFG 双分支把 KV cache 翻倍，设 `cfg_scale=1.0`；或调小 `vae_core_frames` |
| `AbcError: duration ... != meter duration` | 合并时没算上 `'` / `,` 八度记号，小节时长算短了 |
| `Session '<name>' not found` | 那条 assignment 不是 CLI 创建的，接管不了 |
| 授权后仍要重新登录 | 授权页没勾全 scope，或 tmux 里 CLI 进程退出了（用常驻 bash 的 tmux） |

## 已知限制

- **FP16 不是官方验证路径**，与 BF16 的音质等价性未证明，长曲目可能出现数值问题。
- 本仓库跑出来的音频**不能**拿去对标官方 WildSongBench 分数。
- 免费档会话空闲会被后端回收，没有 keep-alive 守护进程（CLI 已移除）。
- 想稳定出活，建议直接上 Colab 付费档的 **L4**（24GB, sm_89）或 A100，删掉全部补丁即可原生运行。

## 目录结构

```
scripts/
  00_probe.py            环境探测（GPU / 内存 / torch / bf16）
  01_setup.py            远端后台 clone + pip install .
  02_patch_fp16.py       BF16 -> FP16（含 gate 绕过），同时打 site-packages 和源码
  03_patch_attention.py  FlashAttention / cuDNN -> SDPA
  04_generate_staged.py  分阶段生成 + 解码，规避 12GB OOM
  05_analyze.py          音频客观体检（可传任意音频路径）
  10_setup_sheetsage.sh  独立 venv 安装 SheetSage2 转谱环境
  11_merge_instrumental.py  Vocal 旋律搬到 Ins，合并成纯音乐谱
  12_validate_abc.py     用官方 abc_tools 校验合并后的谱
  13_generate_cover.py   旋律条件 + 交响风格生成（含长曲关 CFG / 分阶段解码）
sample/
  audio.flac             实测产物（56.84s, 48kHz stereo）
  score.abc              模型生成的 D 大调 / 88 BPM 乐谱
  score_snippet.txt      乐谱片段（README 预览用）
```

## 致谢与许可

- 模型与推理代码：[multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)（Apache-2.0，模型另有 MODEL_LICENSE）
- 显存 offload 思路参考社区 PR #174
- 本仓库脚本仅做兼容性适配，不含模型权重。
