"""按转谱旋律生成交响乐编曲（分阶段 + 释放 MoT 后解码，规避 12GB OOM）。"""
import gc
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import soundfile as sf
import torch

from yue2 import YuE2Pipeline

ABC = Path("/content/cover/score/instrumental.abc")
OUT = Path("/content/outputs/symphony")
OUT.mkdir(parents=True, exist_ok=True)

STYLE = (
    "Symphonic orchestra, cinematic orchestral arrangement, instrumental, no vocals, "
    "full string section with sweeping legato, french horns and trumpet fanfare, "
    "woodwind counterlines, timpani and orchestral percussion, harp, "
    "wide dynamic range, dramatic build into a triumphant finale, 200 BPM"
)
request = {
    "style": STYLE,
    "lyrics": "[Intro]\n[Verse]\n[Chorus]\n[Bridge]\n[Outro]",
    "cot": "melody",
    "seed": 164,
    "cfg_scale": 1.0,       # 关闭 CFG 双分支：长曲 KV cache 直接减半
    "abc": ABC.read_text(encoding="utf-8"),
}

pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B", device="cuda", memory_budget_gib=15, vae_core_frames=512
)
start = time.perf_counter()

plan = pipe.plan(**request)
plan.save(str(OUT / "plan"))
(OUT / "score.abc").write_text(plan.abc or "", encoding="utf-8")
print(f"[stage] plan tokens={len(plan.abc_ids)} t={time.perf_counter() - start:.1f}s")

semantic = pipe.generate_semantic(plan)
print(f"[stage] semantic tokens={len(semantic.tokens)} t={time.perf_counter() - start:.1f}s")

latents = pipe.synthesize(semantic)
array = latents.detach().cpu().numpy() if hasattr(latents, "detach") else np.asarray(latents)
np.save(OUT / "latent.npy", array.astype(np.float32))
print(f"[stage] latent={array.shape} t={time.perf_counter() - start:.1f}s")

pipe._model = None
gc.collect()
torch.cuda.empty_cache()
print("[stage] MoT released, decoding VAE...")

audio = pipe.decode(np.load(OUT / "latent.npy"))
sf.write(str(OUT / "audio.flac"), audio, 48000)
duration = np.asarray(audio).shape[0] / 48000
(OUT / "prompt.txt").write_text(STYLE + "\n\n" + json.dumps(
    {"cot": request["cot"], "seed": request["seed"], "truncated": bool(plan.truncated)}, ensure_ascii=False),
    encoding="utf-8")
print(f"[done] audio={np.asarray(audio).shape} duration={duration:.1f}s "
      f"truncated={plan.truncated} total={time.perf_counter() - start:.1f}s")
