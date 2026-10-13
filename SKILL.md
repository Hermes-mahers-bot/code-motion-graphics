---
name: code-motion-graphics
description: "Use when making an animated ad/video with code."
version: 1.1.0
author: Hermes Agent
---

# Code-rendered motion graphics (animated ads, promos, kinetic type)

Build a 30-90 s animated ad entirely from code: frames drawn with PIL/numpy, score synthesised
with numpy, voice from edge-tts, assembled by ffmpeg. No manim, no Blender, no GPU, no stock
footage, no music licensing. Everything is a variable in a script, so a client note ("change the
flavour, drop the claim, Arabic VO") is a 2-minute edit.

Measured case: 64 s, 1920x1080, 30 fps, 8 shots, 27 cuts, 11 VO lines, original score + foley.
~35 min end to end including two visual revision passes; the render itself is ~5 min on 4 vCPU.

Use it when: product ads, promos, explainer intros, teasers, kinetic-typography, social verticals,
"make me a video for X" where X is a brand/idea rather than real footage.
Do NOT use it when: photoreal footage, actors, real locations, hand-drawn character animation, or
lip sync are needed. That is real footage + ffmpeg. 3D goes to blender-headless-rendering.

## 0. Set expectations honestly

- What comes out is motion-graphics: shapes, gradient, kinetic type, coded product renders.
  It looks like a real edit, not like a film. Say that up front if the ask implies live action.
- 2D coded animation is cheap; AI video generation is not. On a 16 GB iGPU box local gen video is
  out, this pipeline runs fine (PIL is CPU only). Do not reach for ComfyUI for a 60 s ad.
- Two passes are normal: render a stills sheet, fix, then render the film.

## 1. Lock the timing model BEFORE drawing anything

Voice first, music second, pictures third. This ordering is what makes cuts land.

- Pick a BPM and commit: `BPM=120 -> BEAT=0.5s, BAR=2.0s`. Shots of 4 bars = 8.0 s. 8 shots = 64 s.
- Every shot length must be a whole number of bars, and internal cuts on the beat/bar grid, or the
  film feels random however pretty the frames are.
- Write the VO lines per shot, generate them, MEASURE them, then place them:
  `VO_OFFSET = 0.55 s` into each shot (a beat of picture before the voice starts).
- VO density: one line per shot plus 2-3 short punch-ins ("Ice cold.", "Ki Ki!") to fill beats
  where the line ends early. 8 shots x 8 s carrying only 28 s of speech is normal and fine.
- Save `vo/timing.json` (`{key: {file, start, dur}}`) and let the audio mixer read it. Never
  hand-copy timings into two files.
- Keep a cut table in the renderer (`CUTS = [(time, label), ...]`) and use it for the stills review,
  so the storyboard and the film cannot drift apart.

## 2. Shot = ordered beats, each beat a draw function

```python
def beat(t, beats):
    """beats = [(dur, fn, transition)] -> (local_u, fn, transition, index, t_in_cut)"""
    acc = 0.0
    for i, (d, fn, tr) in enumerate(beats):
        if t < acc + d or i == len(beats) - 1:
            return clamp((t - acc) / d), fn, tr, i, (t - acc)
        acc += d
    return 1.0, beats[-1][1], beats[-1][2], len(beats) - 1, 0.0

BEATS = [(3.9, s2a, "cut"), (2.1, s2b, "flash"), (2.0, s2c, "whip")]
u, fn, tr, i, ti = beat(t, BEATS)
frame = trans(fn(u, 0), ti, tr)     # ti = seconds into this cut, drives the transition
```

A 64 s ad wants ~25-30 cuts total (~2.2 s average). Faster than that is unwatchable; slower feels
like a slideshow even if every frame is beautiful.

## 3. Renderer kit (copy these; they are the load-bearing parts)

```python
W, H, FPS = 1920, 1080, 30
from functools import lru_cache
import math, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

POP = "/home/hermes/.fonts/Poppins-ExtraBold.ttf"   # static display face for wordmarks
INT = "/home/hermes/.fonts/Inter.ttf"               # variable: weights via set_variation_by_name
MONO = "/home/hermes/.fonts/JetBrainsMono.ttf"
# check what a box has: fc-list | grep -iE "inter|poppins|montserrat|dejavu"

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out(t): return 1 - (1 - clamp(t)) ** 3
def ease_io(t):  t = clamp(t); return t * t * (3 - 2 * t)
def ease_back(t, s=1.9): t = clamp(t); return 1 + (s+1)*(t-1)**3 + s*(t-1)**2
def lerp(a, b, t): return a + (b - a) * t
def mixc(c1, c2, t): return tuple(int(lerp(c1[i], c2[i], clamp(t))) for i in range(3))

# --- text: cached, tightly cropped, optional letter tracking (ads need tracking)
@lru_cache(maxsize=64)
def _font(path, size, weight):
    f = ImageFont.truetype(path, int(size))
    if weight:
        try: f.set_variation_by_name(weight)      # Inter/Montserrat/JetBrains are variable
        except Exception: pass                     # static fonts (Poppins-ExtraBold) ignore this
    return f

@lru_cache(maxsize=512)
def text_img(text, path, size, weight, fill, stroke=None, sw=0, tracking=0.0):
    f = _font(path, size, weight)
    pad = int(max(sw * 2 + 8, size * 0.12))
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    if tracking:
        wsum = sum(f.getlength(c) + tracking for c in text) - tracking
        asc, desc = f.getmetrics()
        w, h = int(wsum) + 2 * pad, asc + desc + 2 * pad
    else:
        bb = probe.textbbox((0, 0), text, font=f)
        w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    img = Image.new("RGBA", (max(w, 1), max(h, 1)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    st = (stroke + (255,)) if stroke else None
    if tracking:
        x = pad
        for c in text:
            d.text((x, pad), c, font=f, fill=fill, stroke_width=sw, stroke_fill=st)
            x += f.getlength(c) + tracking
    else:
        d.text((pad, pad), text, font=f, fill=fill, stroke_width=sw, stroke_fill=st)
    return img

# --- resize cache. ALWAYS pass an explicit key for per-frame images (see pitfalls)
_RS = {}
def scaled(key, img, w, h):
    w, h = max(1, int(w)), max(1, int(h))
    k = (key, w, h)
    if k not in _RS:
        _RS[k] = img.resize((w, h), Image.LANCZOS)
        if len(_RS) > 4000: _RS.clear()
    return _RS[k]

def blit(base, img, cx, cy, scale=1.0, rot=0.0, alpha=1.0, key=None):
    """alpha-composite img centred on (cx,cy). rot in degrees, only for organic objects."""
    if alpha <= 0.004 or scale <= 0.002: return
    tw, th = int(img.size[0] * scale), int(img.size[1] * scale)
    if tw < 1 or th < 1: return
    im = scaled((key if key is not None else id(img), rot), img, tw, th)
    if abs(rot) > 0.4: im = im.rotate(rot, resample=Image.BILINEAR, expand=True)
    if alpha < 0.999:
        im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(im, (int(cx - im.size[0] / 2), int(cy - im.size[1] / 2)))

def txt(base, text, cx, cy, size, color, font=POP, weight=None, stroke=None, sw=0,
        tracking=0.0, pop=1.0, alpha=1.0, rot=0.0, shadow=None, shadow_off=(0, 0)):
    """pop 0..1 = scale-in progress (the 1.35->1.0 ease is the ad look); rot is in-plane only."""
    sc = 1.0 if pop >= 1 else lerp(1.35, 1.0, ease_out(pop))
    args = (text, font, int(size), weight, tuple(color), stroke, sw, tracking)
    if shadow:
        sh = (text, font, int(size), weight, tuple(shadow), stroke, sw, tracking)
        blit(base, text_img(*sh), cx + shadow_off[0], cy + shadow_off[1], sc, rot, alpha * 0.75)
    blit(base, text_img(*args), cx, cy, sc, rot, alpha)

# --- backgrounds / light. radial() is a cached numpy gradient: the best-looking primitive
@lru_cache(maxsize=16)
def radial(diam, cin, cout, power=1.0):
    y, x = np.mgrid[0:diam, 0:diam]; c = (diam - 1) / 2
    a = np.clip(1 - np.sqrt((x - c) ** 2 + (y - c) ** 2) / c, 0, 1) ** power
    out = np.zeros((diam, diam, 4), np.uint8)
    for i in range(3): out[..., i] = (cin[i] * a + cout[i] * (1 - a)).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")

@lru_cache(maxsize=32)
def gradimg(w, h, c1, c2, horiz=False):
    r = (np.linspace(0, 1, w)[None, :] if horiz else np.linspace(0, 1, h)[:, None])
    arr = np.zeros((h, w, 3), np.uint8)
    for i in range(3): arr[..., i] = (c1[i] + (c2[i] - c1[i]) * r).astype(np.uint8)
    return Image.fromarray(arr, "RGB")

def glow(base, cx, cy, radius, color, alpha=1.0, power=1.6):
    d = 700; blit(base, radial(d, color, (0, 0, 0), power), cx, cy, radius * 2 / d, 0, alpha)

# --- finish: vignette + grain is what makes code art look like a film still
@lru_cache(maxsize=1)
def vignette():
    d = 1600; y, x = np.mgrid[0:d, 0:d]; c = (d - 1) / 2
    a = np.clip((np.sqrt((x - c) ** 2 + (y - c) ** 2) / c - 0.42) / 0.58, 0, 1) ** 1.7
    out = np.zeros((d, d, 4), np.uint8); out[..., 3] = (a * 190).astype(np.uint8)
    return Image.fromarray(out, "RGBA").resize((W, H), Image.BILINEAR)

@lru_cache(maxsize=6)
def grain(seed):
    r = np.random.default_rng(seed)
    a = r.normal(128, 26, (H // 2, W // 2)).astype(np.uint8)
    return Image.fromarray(np.stack([a, a, a, np.full_like(a, 22)], -1), "RGBA").resize((W, H))

def finish(f, seed=0):
    f.alpha_composite(vignette()); f.alpha_composite(grain(seed % 6))
    return f.convert("RGB")
```

Frame loop pipes straight into ffmpeg (never 1900 PNG files):

```python
def render_range(f0, f1, out):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
        "-video_track_timescale", "30000", "-g", "60", out], stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(render_frame(i / FPS).tobytes())
    p.stdin.close(); p.wait()
```

## 4. THE COOL FACTOR COOKBOOK (ordered by impact per line of code)

1. **Landing impact on the hero object.** Expanding rings + dust kick + the contact shadow
   squashing with the fall. This single moment sells the whole ad.
   ```python
   if u > 0.56:                       # after touchdown
       p = clamp((u - 0.56) * 3.2)
       burst_ring(f, W/2, groundY, 200 + 900 * ease_out(p), 16 * (1-p) + 3, YELLOW, int(190*(1-p)))
       burst_ring(f, W/2, groundY, 120 + 620 * ease_out(p), 10 * (1-p) + 2, WHITE, int(150*(1-p)))
   ```
2. **Kill dead frames.** Every transition must resolve into a composed frame with the product or a
   headline in it. A 0.55 s dark beat is fine ONLY if a light beam, glow or type is in it.
3. **Fake yaw spin, never in-plane rotation.** Rotating flat 2D art 360 degrees reads as tumbling.
   Squash the width to fake rotation around the vertical axis, and sweep a highlight across it:
   ```python
   sx = max(0.26, abs(math.cos(2 * math.pi * turn)))
   ci = can_img(); csq = ci.resize((max(2, int(ci.size[0] * sx)), ci.size[1]), Image.BILINEAR)
   blit(f, csq, W/2, H*0.50, 0.44 * (1300/1500), 0, 1.0, key=("can_sq", round(sx, 3)))
   ```
4. **Scrim behind every headline.** Big type over a busy background is unreadable; blur a dark band
   behind it: `fill=(8,10,14,205)` then `GaussianBlur(26)`. Instant legibility.
5. **Kinetic type pop.** `pop` scale 1.35 -> 1.0 with `ease_out`, plus stroke and an offset shadow.
   `tracking=6..12` on uppercase ad type makes it look designed instead of defaulted.
6. **Transitions that cost nothing:** `flash` (white overlay fading over 0.2 s), `whip` (offset the
   outgoing frame, lay 3 alpha copies behind a bright streak), `zoom` (1.22 -> 1.0 crop punch).
   One per cut maximum, and only in the first 0.3 s.
7. **Ray fan behind the hero** (polygon wedges from the object centre, alpha 0.07-0.13, slow
   rotation) plus `sparkles` (crosses with a sine twinkle). Sells "hero frame" instantly.
8. **Parallax layers.** Two skyline bands scrolled at different speeds, a sky gradient, a sun `glow`
   and 3-4 bird arcs. 30 lines for a whole scene.
9. **Sound-reactive bars** on the beat where the VO says a sound word ("crackle"). Seeded random
   heights animated with `abs(sin(t*6.5 + phase))` look like real audio.
10. **Particle systems, seeded, deterministic:** droplets (ellipse + highlight dot), bubbles
    (outlined circles), splash crown (polys above the liquid level), impact dust. Use
    `np.random.default_rng(seed)` so frames are reproducible for the stills pass.
11. **Beat-locked zoom punch / camera shake.** `cam_shake(seed,t,amp=14,freq=24)` on each bar's hit,
    a 6 % zoom on the drop.
12. **Film finish:** vignette + grain (section 3), plus optional chromatic offset on hits:
    ```python
    def chroma(img_rgb, px=6):
        a = np.array(img_rgb)
        a[..., 0] = np.roll(a[..., 0], px, axis=1)
        a[..., 2] = np.roll(a[..., 2], -px, axis=1)
        return Image.fromarray(a)
    ```
13. **Circle/wipe mask transitions** need the previous frame kept: `prev.copy()`, then
    `out.paste(new, (0,0), circle_mask(p).filter(GaussianBlur(18)))`.
14. **Real product geometry beats clip art.** Draw the can/box/bottle from primitives: body gradient
    masked by a rounded rect, label band, stacked wordmark, specular highlight stripes (blurred white
    rounded rects), condensation dots, rim + tab ellipses. ~60 lines, and it reads as the product
    from 3 m away.
15. **Liquid with a real surface:** fill polygon with a sine-wave top edge, a lighter overlay polygon
    at ~70 alpha, ice cubes blitted on top, droplets running down the glass.

## 5. Audio: synthesise it, do not hunt for it

Voice (edge-tts, free, network):

```bash
edge-tts --voice=en-US-EmmaMultilingualNeural --rate=+14% --pitch=+3Hz \
  --text="Brewed real tea. Poured over real ice. No shortcuts." --write-media s3.mp3
```

- 403s come and go per voice: retry 3-4x with backoff, then fall back to another voice
  (`en-US-EmmaMultilingualNeural`, `en-US-AndrewMultilingualNeural` were reliable when
  `AriaNeural`/`GuyNeural` 403'd). Use the `--flag=value` form.
- edge-tts pads lead/tail silence: decode to f32, trim at `|x| > 0.006` with 60 ms padding, or the
  VO lands late by 150-400 ms.
- VO chain: `highpass=f=95,acompressor=threshold=-18dB:ratio=3:attack=5:release=140:makeup=2,`
  `equalizer=f=3200:t=q:w=1.1:g=2.5,equalizer=f=7500:t=q:w=1.5:g=-2.5,loudnorm=I=-16:TP=-1.5:LRA=7`

Score (numpy, 120 BPM, all original so no licensing):

- Primitives: kick (sine, 130->47 Hz pitch envelope), clap (noise + a differenced copy), hat
  (differenced noise, 18 ms decay), bass (sine + 2nd/3rd harmonic), pluck, pad (3 detunes), riser
  (exponential sweep + noise), crash, and foley: can crack, ice clink (3 inharmonic sines), pour
  (band-limited noise with a 3 Hz wobble), crackle (26 random micro-clicks), bubble, bell arp.
- Arrange on a 32-bar grid, one section per shot; strip drums over one bar range to make the final
  drop land (`place(buf, -kick(), t0)` subtracts a hit you already wrote).
- Duck the bed under the voice: build a gain envelope from `timing.json` (0.26 during VO, 150 ms
  ramps) and multiply the music bus by it.
- Mix discipline (where ads die): music bus ~0.40 of full, VO at 1.15, then `tanh(mix*1.45)*0.74`
  and normalise to -3 dBFS before the 16-bit write. With the bed at 0.7 the voice is buried and no
  EQ saves it.

## 6. Render + assemble

```bash
# 4 chunks on 4 vCPU, boundaries ON shot boundaries so concat -c copy is exact
for i in 0 1 2 3; do
  S=$((i * 480)); E=$(((i + 1) * 480))          # 480 frames = 16 s = 2 shots
  python3 render.py range $S $E chunks/chunk_$i.mp4 &
done; wait
: > chunks/list.txt
for i in 0 1 2 3; do echo "file '/ABS/PATH/chunks/chunk_$i.mp4'" >> chunks/list.txt; done
ffmpeg -v error -y -f concat -safe 0 -i chunks/list.txt -c copy video_silent.mp4

# mux + level in one pass: loudnorm's true-peak ceiling is what stops AAC clipping
ffmpeg -v error -y -i video_silent.mp4 -i mix/audio.wav -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -ar 44100 -ac 2 \
  -af "loudnorm=I=-16:TP=-2.5:LRA=9" -movflags +faststart -shortest ad.mp4
```

Cost measured: 0.73 s/frame single-threaded at 1080p -> ~23 min for 64 s, so parallelise.
Sizes for flat graphic content at 1080p30: crf 19 -> ~88 MB/64 s (master), crf 25 -> 12 MB,
1280x720 crf 23 -> 6.4 MB. Chat platforms cap around 50 MB: ship the crf 25 copy, keep the master.

## 7. Verify, or you will ship garbage

1. **Stills pass first**: a `stills` mode that renders one frame per cut into a captioned contact
   sheet. Look at it. This is where clipped text, dead frames, missing product and colour problems
   get caught. It costs 20 s; the full render costs 20 min.
2. **Sample stills MID-BEAT, not at the cut time.** Sampling at `cut_time` renders the transition at
   t_in=0, i.e. a full-white flash or a whip offset, and you will "fix" problems that do not exist.
   Use `cut + 0.45 * beat_len` (min 0.35 s).
3. `ffprobe` the output for `duration`, `nb_frames`, `avg_frame_rate`, and check the frame count
   equals `TOTAL * FPS` exactly (1920 frames for 64 s at 30 fps).
4. **VO audibility table:** decode the final mp4, print RMS dB per VO window vs a control window just
   after it. Aim for >= 4 dB separation; below that, drop the music bus or deepen the duck.
5. Silent-stretch check: no gap longer than ~0.8 s above -45 dBFS. An ad must never go quiet.
6. `volumedetect` + `loudnorm=print_format=summary` on the final file: mean about -19 dB, peak
   between -1 and -4 dB. Anything at 0.0 dB is clipping, however the waveform looks.
7. Pull 8-10 frames out of the FINAL mp4 (not the renderer) and eyeball them. Only this catches an
   encode or mux problem.

## 8. Pitfalls that actually cost time

- **concat demuxer resolves list paths relative to the list file.** A list inside `chunks/` holding
  `chunks/chunk_0.mp4` fails with `Impossible to open 'chunks/chunks/chunk_0.mp4'`. Use absolute paths.
- **Lossy AAC overshoots the source peak.** A master at -3.2 dBFS decoded at 0.0 dBFS (clipped). Do
  not resample 44.1 -> 48 kHz on the way in, and always cap with `loudnorm ... TP=-2.5`. Verify with
  `volumedetect` on the muxed file, not on the WAV.
- **`blit` resize cache keyed on `id(img)` is a landmine** when images are built per frame: CPython
  reuses ids and you can be served a cached resize of a different image. Pass `key=(...)` for any
  per-frame image, or `lru_cache` the generator so the object stays alive.
- **Never rotate flat art in-plane to fake a spin** (cookbook 3).
- **Text at pop=0 is scaled 1.35** and will overlap whatever sits below it. Space hero text for the
  scaled-up state, or move the can up (`cy = H*0.46`) rather than pushing the caption down.
- **Delayed alpha on the money line** (e.g. `alpha = clamp(u*2.4 - 1.0)`) makes the headline land in
  the last 0.4 s of the beat. Headlines must be readable for most of the shot: pop them by `u*3`.
- Compile-check before a long render: `python3 -m py_compile render.py && python3 render.py stills`.
  A missing unpack value (`W, H, FPS = 1920, 1080`) costs 20 s this way, 20 min the other way.
- Long inline shell/python one-liners hit the terminal guard: put code in a `.py`/`.sh` file instead.
- Music bed too loud is the #1 mix failure and invisible until measured (section 7.4).
- "More energy" is almost always solved by cutting, not drawing: shorten beats, add a punch-in line,
  add a flash on the bar. Do not add more elements to an already busy frame.

## 9. Deliverable packaging

- `ad.mp4` (1080p crf 19 master), `ad_deliver.mp4` (crf 25, chat), `ad_720p.mp4` (fallback),
  `poster.jpg` (`-ss <hero time> -frames:v 1 -q:v 3`).
- Send the video with a `MEDIA:` path (no spaces) and list the shot structure in the message so the
  client can point at "shot 4" instead of "the part with the ice".
- State the creative decisions made unilaterally (flavour, claims, wording) and flag any claim that
  must be true before publication. Then offer: shorter cut (30 s / 15 s), other voice, localisation
  (an Arabic VO is one edge-tts voice change plus a text file), or a 9:16 vertical rerender at
  1080x1920 with recentred coordinates.

## 10. Reference implementation

A full working ad built with this skill is in `example/kiki-iced-tea/` of this repo:
`render.py` (shots, kit, stills + range modes), `vo.py`, `score.py`, `render_all.sh`,
`mix_and_mux.sh`, `verify.py`, `contact_sheet.jpg`. Read `render.py` before writing a new ad; the
section 3 kit is lifted from it. Public repo: https://github.com/Hermes-mahers-bot/code-motion-graphics
