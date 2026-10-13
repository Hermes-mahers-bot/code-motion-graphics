"""End-to-end verification: audio level map (is the VO actually present and on top?)
plus a shot-by-shot frame sheet pulled from the FINAL mp4."""
import json
import subprocess
import numpy as np

SR = 44100


def load(path, rate=SR, ch=1):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                        "-ac", str(ch), "-ar", str(rate), "-"], check=True,
                       capture_output=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


HERE = os.path.dirname(os.path.abspath(__file__))
AD = os.environ.get("AD_MP4", os.path.join(HERE, "kiki_ad.mp4"))
x = load(AD, SR, 1)
print(f"decoded final audio: {len(x)/SR:.2f}s  peak {np.max(np.abs(x)):.3f}")

timing = json.load(open(os.path.join(HERE, "vo", "timing.json")))
print("\n  VO window            RMS dB   music-only RMS dB   margin")
for k in sorted(timing, key=lambda k: timing[k]["start"]):
    info = timing[k]
    a, b = int(info["start"] * SR), int((info["start"] + info["dur"]) * SR)
    vo = 20 * np.log10(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-9)
    c = int((info["start"] + info["dur"] + 0.35) * SR)
    d = c + (b - a)
    ctrl = 20 * np.log10(np.sqrt(np.mean(x[c:d] ** 2)) + 1e-9)
    print(f"  {k:>3} {info['start']:5.2f}s {info['dur']:4.2f}s   {vo:7.2f}   "
          f"{ctrl:17.2f}   {vo - ctrl:+6.2f}")

# silence check: no dead air longer than 0.8s anywhere
w = int(0.25 * SR)
rms = np.array([np.sqrt(np.mean(x[i:i + w] ** 2)) for i in range(0, len(x) - w, w)])
thr = 10 ** (-45 / 20)
run = best = 0
for v in rms:
    run = run + 1 if v < thr else 0
    best = max(best, run)
print(f"\nlongest silent stretch: {best * 0.25:.2f}s  (threshold -45dBFS)")

# frame sheet from the final file, one settled frame per shot
times = [4.9, 10.2, 19.5, 25.8, 36.4, 41.8, 52.9, 62.9]
for i, t in enumerate(times):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i",
                    AD, "-frames:v", "1",
                    f"{HERE}/check/final_{i+1}.png"], check=True)
print("extracted", len(times), "frames from the final mp4")
