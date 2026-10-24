"""分阶段生成：规避免费档 12.7GB 系统内存的 OOM（Exit 137）。

官方一条龙 pipeline(**request) 会在「Loading audio decoder」处被 OOM Killer 杀掉：
decode() 先把 MoT 挪到 CPU，再加载 VAE 解码器，7.2GB + 解码器同时占内存。

这里拆成：
    plan -> generate_semantic -> synthesize -> 存 latent
    -> 释放 MoT -> 单独 decode(latent)

latent 落盘后，换解码参数 / 换 decoder 不需要重新生成。
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from yue2 import YuE2Pipeline

OUT = Path("/content/outputs/song1")
OUT.mkdir(parents=True, exist_ok=True)
request = json.loads(Path("/content/YuE/examples/song.json").read_text(encoding="utf-8"))

# memory_budget_gib=12 适配 16GB 卡；vae_core_frames=512 降低解码峰值
pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B", device="cuda", memory_budget_gib=12, vae_core_frames=512
)
start = time.perf_counter()

plan = pipe.plan(**request)
if plan.abc:
    (OUT / "score.abc").write_text(plan.abc, encoding="utf-8")
plan.save(str(OUT / "plan"))
print(f"[stage] plan tokens={len(plan.abc_ids)} t={time.perf_counter() - start:.1f}s")

semantic = pipe.generate_semantic(plan)
print(f"[stage] semantic tokens={len(semantic.tokens)} t={time.perf_counter() - start:.1f}s")

latents = pipe.synthesize(semantic)
array = latents.detach().cpu().numpy() if hasattr(latents, "detach") else np.asarray(latents)
np.save(OUT / "latent.npy", array.astype(np.float32))
print(f"[stage] latent={array.shape} t={time.perf_counter() - start:.1f}s")

# 关键一步：释放 MoT，给 VAE 解码腾出系统内存
pipe._model = None
gc.collect()
torch.cuda.empty_cache()
print("[stage] MoT released, decoding VAE...")

audio = pipe.decode(np.load(OUT / "latent.npy"))
sf.write(str(OUT / "audio.flac"), audio, 48000)
print(f"[done] audio={np.asarray(audio).shape} total={time.perf_counter() - start:.1f}s -> {OUT / 'audio.flac'}")
