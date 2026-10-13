"""Ki Ki ad - voiceover generation + timing.
Generates each shot's VO with edge-tts, trims silence, light broadcast processing,
and writes vo/timing.json with per-shot start/duration for the audio mixer.
"""
import json
import os
import subprocess
import time
import numpy as np

SR = 44100
VOICE = "en-US-EmmaMultilingualNeural"
FALLBACK = "en-US-AndrewMultilingualNeural"
RATE = "+14%"
PITCH = "+3Hz"
VO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vo")

# shot index -> line. Each shot is 8.0s on the timeline, VO starts 0.6s into the shot.
LINES = {
    1: "It's hot out. You're thirsty. And everything in the fridge is boring.",
    2: "So we made something better. Meet Ki Ki. Iced tea, done right.",
    3: "Brewed real tea. Poured over real ice. No shortcuts. No powder.",
    4: "Listen. That crackle? That's the whole point.",
    5: "Peach and lemon, with a little sparkle at the end.",
    6: "Made for rooftops, road trips, and long afternoons that never end.",
    7: "Less sugar. No fake syrup. Just cold, clean flavor.",
    8: "Ki Ki. Iced tea. Keep it cold, keep it close.",
}

SHOT_LEN = 8.0
VO_OFFSET = 0.55

# extra punch-ins so every shot has a voice (key -> absolute start seconds, text)
EXTRA = {"2b": (14.5, "Ki Ki!"), "4b": (30.4, "Ice cold."), "5b": (38.4, "Ki Ki!")}


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def gen_mp3(text, out):
    """edge-tts 403s intermittently - retry, then fall back to the other voice."""
    last = None
    for voice in (VOICE, FALLBACK):
        for attempt in range(4):
            p = subprocess.run(["edge-tts", f"--voice={voice}", f"--rate={RATE}",
                                f"--pitch={PITCH}", f"--text={text}",
                                f"--write-media={out}"], capture_output=True)
            if p.returncode == 0 and os.path.getsize(out) > 4000:
                if voice != VOICE:
                    print(f"  (note: used fallback voice {voice})")
                return
            last = p.stderr.decode()[-300:]
            time.sleep(1.5 + attempt * 1.5)
    raise RuntimeError(f"edge-tts failed: {last}")


def load_f32(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
                        "-ac", "1", "-ar", str(SR), "-"], check=True,
                       capture_output=True)
    return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)


def trim(x, thresh=0.006, pad=0.06):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(pad * SR))
    b = min(len(x), idx[-1] + int(pad * SR))
    return x[a:b]


def write_wav(x, path):
    raw = x.astype("<f4").tobytes()
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ac", "1",
                        "-ar", str(SR), "-i", "-", "-c:a", "pcm_s16le", path],
                       input=raw, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode())


def process(src, dst):
    """broadcast-ish VO chain: highpass, compressor, presence, de-ess-ish shelf."""
    af = ("highpass=f=95,acompressor=threshold=-18dB:ratio=3:attack=5:release=140:makeup=2,"
          "equalizer=f=3200:t=q:w=1.1:g=2.5,equalizer=f=7500:t=q:w=1.5:g=-2.5,"
          "loudnorm=I=-16:TP=-1.5:LRA=7")
    run(["ffmpeg", "-v", "error", "-y", "-i", src, "-af", af,
         "-c:a", "pcm_s16le", dst])


def main():
    os.makedirs(VO_DIR, exist_ok=True)
    timing = {}
    for i, line in LINES.items():
        raw_mp3 = f"{VO_DIR}/s{i}_raw.mp3"
        gen_mp3(line, raw_mp3)
        x = trim(load_f32(raw_mp3))
        write_wav(x, f"{VO_DIR}/s{i}_trim.wav")
        process(f"{VO_DIR}/s{i}_trim.wav", f"{VO_DIR}/s{i}.wav")
        dur = len(x) / SR
        start = (i - 1) * SHOT_LEN + VO_OFFSET
        timing[str(i)] = {"file": f"{VO_DIR}/s{i}.wav", "start": round(start, 3),
                          "dur": round(dur, 3)}
        print(f"shot {i}: vo {dur:.2f}s  start {start:.2f}s  end {start+dur:.2f}s  | {line[:40]}")
    for key, (start, text) in EXTRA.items():
        raw_mp3 = f"{VO_DIR}/{key}_raw.mp3"
        gen_mp3(text, raw_mp3)
        x = trim(load_f32(raw_mp3))
        write_wav(x, f"{VO_DIR}/{key}_trim.wav")
        process(f"{VO_DIR}/{key}_trim.wav", f"{VO_DIR}/{key}.wav")
        dur = len(x) / SR
        timing[key] = {"file": f"{VO_DIR}/{key}.wav", "start": round(start, 3),
                       "dur": round(dur, 3)}
        print(f"extra {key}: vo {dur:.2f}s start {start:.2f}s | {text}")
    with open(f"{VO_DIR}/timing.json", "w") as f:
        json.dump(timing, f, indent=1)
    print("total timeline:", len(LINES) * SHOT_LEN, "s")


if __name__ == "__main__":
    main()
