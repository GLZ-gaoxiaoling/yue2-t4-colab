"""对生成音频做客观体检，确认不是静音 / 噪声 / 削波。

用法：python 05_analyze.py [/content/outputs/song1/audio.flac]
"""
import sys

import numpy as np
import soundfile as sf

path = sys.argv[1] if len(sys.argv) > 1 else "/content/outputs/song1/audio.flac"
audio, rate = sf.read(path)
mono = audio.mean(axis=1)
duration = len(mono) / rate
rms = np.sqrt((mono ** 2).mean())
peak = np.abs(mono).max()

print(f"sample_rate={rate} channels={audio.shape[1]} duration={duration:.2f}s")
print(f"rms={20 * np.log10(max(rms, 1e-9)):.2f} dBFS peak={20 * np.log10(max(peak, 1e-9)):.2f} dBFS")
print(f"clipping_ratio={(np.abs(mono) > 0.999).mean():.5f}")

frame = rate * 8
for offset in range(0, len(mono), frame):
    chunk = mono[offset:offset + frame]
    if len(chunk) < rate // 2:
        continue
    level = 20 * np.log10(max(np.sqrt((chunk ** 2).mean()), 1e-9))
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
    freqs = np.fft.rfftfreq(len(chunk), 1 / rate)
    centroid = float((spectrum * freqs).sum() / max(spectrum.sum(), 1e-9))
    print(f"  [{offset / rate:5.1f}-{(offset + len(chunk)) / rate:5.1f}s] rms={level:6.2f} dB centroid={centroid:7.1f} Hz")
