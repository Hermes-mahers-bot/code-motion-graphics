"""Ki Ki ad - music + foley score, 64s @ 120 BPM, built from synth primitives.
Writes mix/music.wav (music+foley+VO with sidechain ducking) and mix/video_audio.wav ready to mux.
"""
import json
import os
import subprocess
import numpy as np

SR = 44100
BPM = 120
BEAT = 60.0 / BPM          # 0.5s
BAR = 4 * BEAT             # 2.0s
SHOT = 8.0
NSHOTS = 8
TOTAL = NSHOTS * SHOT      # 64s
N = int(TOTAL * SR)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mix")
VO_DIR = os.path.join(HERE, "vo")

rng = np.random.default_rng(7)


def nz(n):
    return rng.normal(0, 1, n).astype(np.float32)


def env_exp(n, attack=0.002, decay=0.25):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-5), 0, 1)
    return (a * np.exp(-t / decay)).astype(np.float32)


def place(buf, sig, t):
    i = int(t * SR)
    if i < 0:
        sig = sig[-i:]
        i = 0
    j = min(len(buf), i + len(sig))
    if j <= i:
        return
    buf[i:j] += sig[:j - i]


def sine(f, n, phase=0.0):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t + phase).astype(np.float32)


def kick(dur=0.30):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 130 * np.exp(-t / 0.035) + 47
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.10)
    click = nz(int(0.004 * SR)) * np.linspace(1, 0, int(0.004 * SR))
    out = body.astype(np.float32) * 0.95
    place(out, click * 0.35, 0)
    return out


def clap(dur=0.22):
    n = int(dur * SR)
    base = nz(n) * env_exp(n, 0.001, 0.055)
    tail = nz(n) * env_exp(n, 0.02, 0.11)
    sig = base * 0.5 + tail * 0.32
    # bandpass-ish: emphasise 1-3k by differencing then smoothing
    d = np.diff(sig, prepend=0.0)
    sm = np.convolve(d, np.ones(3) / 3, mode="same")
    return (sig * 0.6 + sm * 0.7).astype(np.float32)


def hat(dur=0.05, amp=0.22):
    n = int(dur * SR)
    x = nz(n)
    hp = np.diff(x, prepend=0.0)      # crudely high-passed
    return (hp * env_exp(n, 0.0005, 0.018) * amp * 3).astype(np.float32)


def openhat(dur=0.30, amp=0.16):
    n = int(dur * SR)
    x = np.diff(nz(n), prepend=0.0)
    return (x * env_exp(n, 0.001, 0.10) * amp * 3).astype(np.float32)


def bass(f, dur, amp=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * t
    sig = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    e = np.clip(t / 0.006, 0, 1) * (0.65 + 0.35 * np.exp(-t / 0.20))
    return (sig * e * amp / 1.4).astype(np.float32)


def pluck(f, dur=0.35, amp=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * f * t
    sig = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.10 * np.sin(3 * ph) + 0.05 * np.sin(4 * ph)
    return (sig * env_exp(n, 0.003, 0.09) * amp).astype(np.float32)


def pad(freqs, dur, amp=0.13):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        for d in (1.0, 0.5, 0.25):
            sig += (np.sin(2 * np.pi * f * d * t) * d).astype(np.float32)
    e = np.clip(t / 0.35, 0, 1) * np.clip((dur - t) / 0.45, 0, 1)
    return (sig * e * amp / 3).astype(np.float32)


def riser(dur, f0=200, f1=3200, amp=0.30):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (f1 / f0) ** (t / dur)
    ph = 2 * np.pi * np.cumsum(f) / SR
    sweep = np.sin(ph) * 0.5
    noise = np.diff(nz(n), prepend=0.0) * 0.8
    e = (t / dur) ** 1.6
    return ((sweep + noise) * e * amp).astype(np.float32)


def crash(dur=1.2, amp=0.30):
    n = int(dur * SR)
    x = np.diff(nz(n), prepend=0.0)
    return (x * env_exp(n, 0.002, 0.30) * amp * 2.2).astype(np.float32)


def whoosh(dur=0.32, amp=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = nz(n)
    e = np.exp(-((t - dur * 0.42) ** 2) / (2 * (dur * 0.20) ** 2))
    bp = np.convolve(np.diff(x, prepend=0.0), np.ones(4) / 4, mode="same")
    return (bp * e * amp * 2.0).astype(np.float32)


# ---- foley -------------------------------------------------------------
def can_crack(amp=0.55):
    n = int(0.55 * SR)
    click = nz(int(0.01 * SR)) * 0.7
    body = sine(1180, n) * env_exp(n, 0.001, 0.09) * 0.5
    body2 = sine(1760, n) * env_exp(n, 0.001, 0.06) * 0.25
    out = (body + body2).astype(np.float32)
    place(out, click.astype(np.float32), 0)
    return out * amp * 2.2


def ice_clink(amp=0.30, base=2400):
    n = int(0.30 * SR)
    sig = (sine(base, n) * env_exp(n, 0.0008, 0.055) * 0.5
           + sine(base * 1.51, n) * env_exp(n, 0.0008, 0.035) * 0.28
           + sine(base * 2.13, n) * env_exp(n, 0.0008, 0.022) * 0.16)
    return (sig * amp * 2.4).astype(np.float32)


def pour(dur, amp=0.14):
    n = int(dur * SR)
    x = nz(n)
    bp = np.convolve(np.diff(x, prepend=0.0), np.ones(5) / 5, mode="same")
    t = np.arange(n) / SR
    e = np.clip(t / 0.25, 0, 1) * np.clip((dur - t) / 0.5, 0, 1)
    wob = 0.75 + 0.25 * np.sin(2 * np.pi * 3.1 * t)
    return (bp * e * wob * amp * 2.4).astype(np.float32)


def crackle(dur, events=26, amp=0.16):
    out = np.zeros(int(dur * SR), dtype=np.float32)
    for _ in range(events):
        t = rng.uniform(0, dur - 0.1)
        c = nz(int(0.03 * SR)) * env_exp(int(0.03 * SR), 0.0004, 0.007)
        place(out, (c * rng.uniform(0.4, 1.0) * amp * 2.5).astype(np.float32), t)
    return out


def bell_arp(amp=0.22):
    out = np.zeros(int(1.1 * SR), dtype=np.float32)
    for k, mult in enumerate([1.0, 1.25, 1.5, 2.0]):
        f = 900 * mult
        n = int(0.5 * SR)
        sig = (sine(f, n) * env_exp(n, 0.003, 0.14) * 0.5
               + sine(f * 2.0, n) * env_exp(n, 0.003, 0.08) * 0.2)
        place(out, (sig * amp).astype(np.float32), k * 0.085)
    return out


def bubble(amp=0.22, f=520):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    ff = f * (1 + 2.6 * t / 0.16)
    ph = 2 * np.pi * np.cumsum(ff) / SR
    return (np.sin(ph) * env_exp(n, 0.002, 0.045) * amp).astype(np.float32)


# ---- arrangement -------------------------------------------------------
def music_bed():
    buf = np.zeros(N, dtype=np.float32)
    # chords per 2-bar block (F C G Am in C major-ish, sunny)
    prog = [("F", [174.61, 220.0, 261.63]),
            ("C", [130.81, 196.0, 261.63]),
            ("G", [196.0, 246.94, 293.66]),
            ("A", [220.0, 261.63, 329.63])]
    bass_root = {"F": 43.65, "C": 32.70, "G": 49.0, "A": 55.0}
    scale = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25]

    total_bars = int(TOTAL / BAR)          # 32 bars
    for b in range(total_bars):
        t0 = b * BAR
        chord = prog[(b // 2) % 4]
        name = chord[0]
        # --- drums
        if b >= 1:                          # sparse tension in bar 0
            for k in range(4):
                place(buf, kick() * (0.9 if b >= 4 else 0.7), t0 + k * BEAT)
        if b >= 4:
            for k in (1, 3):
                place(buf, clap(), t0 + k * BEAT)
            for k in range(8):
                place(buf, hat(amp=0.18 if k % 2 else 0.10), t0 + k * BEAT / 2)
            if b % 4 == 3:
                place(buf, openhat(), t0 + 3.5 * BEAT)
        else:
            for k in range(8):
                place(buf, hat(amp=0.07), t0 + k * BEAT / 2)
        # --- bass: 8th note pulse once the beat drops
        if b >= 4:
            for k in range(8):
                amp = 0.42 if k % 2 == 0 else 0.28
                place(buf, bass(bass_root[name], BEAT / 2 * 0.92, amp), t0 + k * BEAT / 2)
        else:
            place(buf, bass(bass_root[name] / 2, BAR * 0.9, 0.34), t0)
        # --- pads during verse-ish sections
        if b >= 4:
            place(buf, pad(chord[1] and [c / 2 for c in chord[1]], BAR, 0.10), t0)
        # --- melody: 16th-ish plucks, denser in later shots
        if b >= 6:
            steps = [0, 2, 3, 5] if b % 4 in (0, 1) else [0, 2, 4, 6, 7]
            for s in steps:
                f = scale[(b * 3 + s) % len(scale)]
                place(buf, pluck(f, 0.32, 0.16), t0 + s * BEAT / 2)
        # --- breakdown before final shot (bars 24-27 = shot 7)
        if b in (24, 25, 26, 27):
            pass
    # breakdown: strip drums for bars 24-26, keep pad + build
    return buf, prog, bass_root, scale


def build():
    buf, prog, bass_root, scale = music_bed()

    # --- section shaping -------------------------------------------------
    # Shot 7 (bars 24-27) breakdown: silence the 4-on-floor, add claps+riser
    bd = int(24 * BAR * SR)
    bd_end = int(28 * BAR * SR)
    # (music_bed already wrote bars>=4 kicks; subtract them in the breakdown)
    for b in (24, 25, 26, 27):
        t0 = b * BAR
        for k in range(4):
            place(buf, -kick() * 0.9, t0 + k * BEAT)
        for k in (1, 3):
            place(buf, -clap(), t0 + k * BEAT)
    for b in (24, 26):
        for k in (1, 3):
            place(buf, clap() * 0.8, b * BAR + k * BEAT)
    place(buf, riser(4.0, 200, 4200, 0.30), 24 * BAR)

    # --- section risers / crashes ---------------------------------------
    place(buf, riser(2.0, 150, 2600, 0.26), 6 * BAR)          # into reveal
    place(buf, crash(1.4, 0.34), 4 * BAR)                     # shot 2 drop hit
    place(buf, crash(1.2, 0.22), 8 * BAR)
    place(buf, crash(1.0, 0.20), 12 * BAR)
    place(buf, crash(1.0, 0.20), 16 * BAR)
    place(buf, riser(2.0, 180, 3000, 0.24), 20 * BAR)
    place(buf, crash(1.6, 0.40), 28 * BAR)                    # final drop
    place(buf, crash(1.8, 0.30), TOTAL - 0.05)

    # final shot: extra energy - 16th hats for the last 8 bars
    for b in range(28, 32):
        t0 = b * BAR
        for k in range(16):
            place(buf, hat(amp=0.10, dur=0.035), t0 + k * BEAT / 4)

    # --- foley -----------------------------------------------------------
    place(buf, can_crack(0.6), 8.05)            # reveal
    place(buf, whoosh(0.30, 0.26), 7.78)        # into reveal
    for k, t in enumerate([0.35, 1.1, 1.7]):
        place(buf, ice_clink(0.30, 2300 + 140 * k), 16.0 + t)
    place(buf, ice_clink(0.34, 2600), 17.9)
    place(buf, pour(5.0, 0.13), 17.60)          # pour runs through shot 3
    place(buf, bubble(0.26, 480), 20.4)
    place(buf, bubble(0.20, 700), 21.6)
    place(buf, can_crack(0.5), 24.10)           # ice macro
    place(buf, crackle(3.6, 30, 0.15), 24.30)
    place(buf, whoosh(0.26, 0.22), 23.80)
    place(buf, bell_arp(0.26), 32.10)           # flavor sparkle
    place(buf, bell_arp(0.20), 34.60)
    place(buf, whoosh(0.26, 0.20), 31.80)
    place(buf, whoosh(0.24, 0.20), 39.80)
    place(buf, ice_clink(0.30, 2500), 40.40)
    place(buf, whoosh(0.22, 0.18), 47.80)
    place(buf, ice_clink(0.28, 2100), 48.30)
    place(buf, bubble(0.24, 560), 49.2)
    place(buf, can_crack(0.5), 56.10)           # CTA reveal
    place(buf, ice_clink(0.32, 2700), 58.05)    # can clink
    place(buf, ice_clink(0.26, 3000), 58.16)
    place(buf, bell_arp(0.30), 62.20)           # outro chime
    place(buf, whoosh(0.28, 0.22), 63.55)

    # --- VO with ducking -------------------------------------------------
    with open(f"{VO_DIR}/timing.json") as f:
        timing = json.load(f)
    vo_bus = np.zeros(N, dtype=np.float32)
    duck = np.ones(N, dtype=np.float32)
    for k, info in timing.items():
        x = load_f32(info["file"])
        i = int(info["start"] * SR)
        j = min(N, i + len(x))
        vo_bus[i:j] += x[:j - i]
        # duck envelope with 150ms ramps
        r = int(0.15 * SR)
        a, b = max(0, i - r), min(N, j + r)
        seg = np.ones(b - a, dtype=np.float32)
        seg[:min(r, len(seg))] = np.linspace(1, 0.26, min(r, len(seg)))
        seg[-min(r, len(seg)):] = np.linspace(0.26, 1, min(r, len(seg)))
        duck[a:b] = np.minimum(duck[a:b], seg)

    music = buf * duck * 0.40          # bed sits ~7dB under the voice
    vo_bus *= 1.15
    mix = music + vo_bus

    # gentle bus glue: soft-knee saturation, then limit and normalise with headroom
    mix = np.tanh(mix * 1.45) * 0.74
    # 3ms look-ahead style peak tame so transients (kick/clap) do not poke through
    peak = float(np.max(np.abs(mix))) or 1.0
    mix *= (10 ** (-3.2 / 20)) / peak
    return mix.astype(np.float32)


def write_wav(x, path, stereo=True):
    if stereo:
        st = np.stack([x, np.roll(x, int(0.0015 * SR))], axis=1).reshape(-1)
        data = np.clip(st, -1, 1).astype("<f4").tobytes()
        ch = "2"
    else:
        data = np.clip(x, -1, 1).astype("<f4").tobytes()
        ch = "1"
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", ch,
                        "-ar", str(SR), "-i", "-", "-c:a", "pcm_s16le", path],
                       input=data, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())


def load_f32(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                        "-ac", "1", "-ar", str(SR), "-"], check=True,
                       capture_output=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mix = build()
    write_wav(mix, f"{OUT}/audio.wav")
    print("wrote", f"{OUT}/audio.wav", "dur", len(mix) / SR, "peak", float(np.max(np.abs(mix))))
