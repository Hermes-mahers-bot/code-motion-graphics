"""Ki Ki Iced Tea - 64s fast-cut 2D animated ad.
Pure PIL/numpy frame renderer piped straight into ffmpeg (no manim, no GPU).
8 shots, 30 internal cuts, kinetic type, coded product renders, particles, camera moves.

usage:
  python3 render.py stills      -> renders one frame per cut to stills/ + contact sheet
  python3 render.py video       -> renders full 64s 30fps to kiki_ad.mp4 (silent)
"""
import math
import os
import subprocess
import sys
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
SHOT = 8.0
NSHOTS = 8
TOTAL = NSHOTS * SHOT
OUTDIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = f"{OUTDIR}/stills"

NAVY = (11, 26, 38)
DEEP = (5, 13, 20)
TEAL = (23, 195, 178)
TEAL_D = (10, 116, 110)
YELLOW = (255, 201, 60)
YELLOW_D = (214, 150, 20)
PEACH = (255, 122, 92)
PEACH_L = (255, 176, 138)
CREAM = (255, 247, 230)
AMBER = (198, 122, 38)
AMBER_L = (245, 186, 88)
AMBER_XL = (255, 214, 140)
ICE = (206, 236, 246)
ICE_D = (120, 190, 215)
WHITE = (255, 255, 255)
GREY = (108, 118, 126)
GREY_D = (58, 66, 72)

POP = os.environ.get("POP_FONT", os.path.expanduser("~/.fonts/Poppins-ExtraBold.ttf"))
INT = os.environ.get("INT_FONT", os.path.expanduser("~/.fonts/Inter.ttf"))
MONO = os.environ.get("MONO_FONT", os.path.expanduser("~/.fonts/JetBrainsMono.ttf"))


# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(t):
    return 1 - (1 - clamp(t)) ** 3


def ease_in(t):
    return clamp(t) ** 3


def ease_io(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_back(t, s=1.9):
    t = clamp(t)
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2


def lerp(a, b, t):
    return a + (b - a) * t


def mixc(c1, c2, t):
    return tuple(int(lerp(c1[i], c2[i], clamp(t))) for i in range(3))


# ---------------------------------------------------------------- fonts / text
@lru_cache(maxsize=64)
def _font(path, size, weight):
    f = ImageFont.truetype(path, int(size))
    if weight:
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
    return f


@lru_cache(maxsize=512)
def text_img(text, path, size, weight, fill, stroke=None, sw=0, tracking=0.0):
    """RGBA text, tightly cropped, optional letter tracking."""
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
    fill = fill + (255,) if len(fill) == 3 else fill
    st = (stroke + (255,)) if (stroke and len(stroke) == 3) else stroke
    if tracking:
        x = pad
        for c in text:
            d.text((x, pad), c, font=f, fill=fill, stroke_width=sw, stroke_fill=st)
            x += f.getlength(c) + tracking
    else:
        d.text((pad, pad), text, font=f, fill=fill, stroke_width=sw, stroke_fill=st)
    return img


_RS = {}


def scaled(key, img, w, h):
    w, h = max(1, int(w)), max(1, int(h))
    k = (key, w, h)
    got = _RS.get(k)
    if got is None:
        got = img.resize((w, h), Image.LANCZOS)
        if len(_RS) > 4000:
            _RS.clear()
        _RS[k] = got
    return got


@lru_cache(maxsize=16)
def radial(diam, cin, cout, power=1.0):
    y, x = np.mgrid[0:diam, 0:diam]
    c = (diam - 1) / 2
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
    a = np.clip(1 - r, 0, 1) ** power
    out = np.zeros((diam, diam, 4), np.uint8)
    for i in range(3):
        out[..., i] = (cin[i] * a + cout[i] * (1 - a)).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


@lru_cache(maxsize=32)
def gradimg(w, h, c1, c2, horiz=False):
    if horiz:
        r = np.linspace(0, 1, w)[None, :]
    else:
        r = np.linspace(0, 1, h)[:, None]
    arr = np.zeros((h, w, 3), np.uint8)
    for i in range(3):
        arr[..., i] = (c1[i] + (c2[i] - c1[i]) * r).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


@lru_cache(maxsize=8)
def vignette():
    d = 1600
    y, x = np.mgrid[0:d, 0:d]
    c = (d - 1) / 2
    r = np.sqrt((x - c) ** 2 + (y - c) ** 2) / c
    a = np.clip((r - 0.42) / 0.58, 0, 1) ** 1.7
    out = np.zeros((d, d, 4), np.uint8)
    out[..., 3] = (a * 190).astype(np.uint8)
    return Image.fromarray(out, "RGBA").resize((W, H), Image.BILINEAR)


@lru_cache(maxsize=6)
def grain(seed):
    r = np.random.default_rng(seed)
    a = r.normal(128, 26, (H // 2, W // 2)).astype(np.uint8)
    g = np.stack([a, a, a, np.full_like(a, 22)], -1)
    return Image.fromarray(g, "RGBA").resize((W, H), Image.BILINEAR)


def blit(base, img, cx, cy, scale=1.0, rot=0.0, alpha=1.0, key=None):
    if alpha <= 0.004 or scale <= 0.002:
        return
    w, h = img.size
    tw, th = int(w * scale), int(h * scale)
    if tw < 1 or th < 1:
        return
    key = (key if key is not None else id(img), rot)
    im = scaled(key, img, tw, th)
    if abs(rot) > 0.4:
        im = im.rotate(rot, resample=Image.BILINEAR, expand=True)
    if alpha < 0.999:
        im = im.copy()
        im.putalpha(im.getchannel("A").point(lambda v: int(v * alpha)))
    base.alpha_composite(im, (int(cx - im.size[0] / 2), int(cy - im.size[1] / 2)))


def txt(base, text, cx, cy, size, color, font=POP, weight=None, stroke=None, sw=0,
        tracking=0.0, pop=0.0, alpha=1.0, rot=0.0, shadow=None, shadow_off=(0, 0)):
    """draw text centred on (cx,cy). pop = 0..1 scale-in progress."""
    sc = 1.0 if pop >= 1 else lerp(1.35, 1.0, ease_out(pop))
    key = (text, font, int(size), weight, color, stroke, sw, round(tracking, 2))
    img = text_img(text, font, int(size), weight, tuple(color), stroke, sw, tracking)
    if shadow:
        sh = text_img(text, font, int(size), weight, tuple(shadow), stroke, sw, tracking)
        blit(base, sh, cx + shadow_off[0], cy + shadow_off[1], sc, rot, alpha * 0.75)
    blit(base, img, cx, cy, sc, rot, alpha)


# ---------------------------------------------------------------- product art
@lru_cache(maxsize=2)
def can_img(seed_h=1500):
    """slim iced tea can, drawn once at high res."""
    h = seed_h
    w = int(h * 0.44)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    r = int(w * 0.20)
    body = gradimg(w, h, AMBER_XL, (176, 104, 30))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    img.paste(body, (0, 0), mask)
    # label band
    ly0, ly1 = int(h * 0.155), int(h * 0.845)
    d.rounded_rectangle([int(w * 0.045), ly0, w - int(w * 0.045), ly1],
                        int(w * 0.06), fill=NAVY + (255,))
    d.rounded_rectangle([int(w * 0.045), ly0, w - int(w * 0.045), ly0 + int(h * 0.006)],
                        int(w * 0.06), fill=TEAL + (255,))
    # label art: KI / KI stacked wordmark
    ws = int(w * 0.42)
    ki = text_img("KI", POP, ws, None, CREAM, None, 0, 0)
    blit(img, ki, w / 2, h * 0.30, 1.0)
    blit(img, ki, w / 2, h * 0.30 + ws * 1.16, 1.0)
    sub = text_img("ICED TEA", INT, int(w * 0.135), "Bold", TEAL, None, 0, 3)
    blit(img, sub, w / 2, h * 0.545, 1.0)
    fl = text_img("PEACH + LEMON", INT, int(w * 0.085), "SemiBold", YELLOW, None, 0, 2)
    blit(img, fl, w / 2, h * 0.63, 1.0)
    # fruit dots
    d.ellipse([w * 0.20, h * 0.685, w * 0.20 + w * 0.16, h * 0.685 + w * 0.16],
              fill=PEACH + (255,))
    d.ellipse([w * 0.62, h * 0.685, w * 0.62 + w * 0.16, h * 0.685 + w * 0.16],
              fill=YELLOW + (255,))
    d.arc([w * 0.62, h * 0.685, w * 0.62 + w * 0.16, h * 0.685 + w * 0.16], 100, 260,
          fill=YELLOW_D + (255,), width=max(2, int(w * 0.02)))
    # top rim + tab
    d.rounded_rectangle([0, 0, w - 1, int(h * 0.055)], int(w * 0.16),
                        fill=(196, 202, 208, 255))
    d.ellipse([w * 0.16, int(h * 0.006), w * 0.84, int(h * 0.05)],
              fill=(150, 158, 166, 255))
    d.ellipse([w * 0.30, int(h * 0.014), w * 0.70, int(h * 0.042)],
              fill=(112, 120, 128, 255))
    d.ellipse([w * 0.575, int(h * 0.019), w * 0.645, int(h * 0.034)],
              fill=(206, 212, 218, 255))   # tab ring
    # bottom shadow band
    d.rounded_rectangle([int(w * 0.03), int(h * 0.955), w - int(w * 0.03), h - 2],
                        int(w * 0.10), fill=(120, 70, 18, 255))
    # specular highlights
    hl = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl, "RGBA")
    hd.rounded_rectangle([w * 0.20, h * 0.02, w * 0.30, h * 0.98], int(w * 0.05),
                         fill=(255, 255, 255, 52))
    hd.rounded_rectangle([w * 0.34, h * 0.03, w * 0.375, h * 0.97], int(w * 0.02),
                         fill=(255, 255, 255, 30))
    hd.rounded_rectangle([w * 0.78, h * 0.04, w * 0.86, h * 0.96], int(w * 0.04),
                         fill=(255, 255, 255, 26))
    img.alpha_composite(hl.filter(ImageFilter.GaussianBlur(4)))
    # condensation
    cd = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdr = ImageDraw.Draw(cd, "RGBA")
    rr = np.random.default_rng(11)
    for _ in range(150):
        x = rr.uniform(0.05, 0.95) * w
        y = rr.uniform(0.03, 0.97) * h
        if 0.045 < x / w < 0.955 and 0.155 < y / h < 0.845 and rr.random() < 0.65:
            continue
        s = rr.uniform(0.006, 0.020) * w
        cdr.ellipse([x, y, x + s, y + s * rr.uniform(1.0, 1.7)],
                    fill=(255, 255, 255, int(rr.uniform(35, 95))))
    img.alpha_composite(cd)
    return img


@lru_cache(maxsize=4)
def ice_img(size=420):
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    r = int(s * 0.16)
    core = radial(s, (250, 255, 255), ICE_D, 0.6)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], r, fill=255)
    img.paste(core, (0, 0), mask)
    d.rounded_rectangle([2, 2, s - 3, s - 3], r, outline=(255, 255, 255, 190),
                        width=max(2, int(s * 0.018)))
    d.polygon([(s * 0.18, s * 0.30), (s * 0.62, s * 0.10), (s * 0.72, s * 0.22),
               (s * 0.26, s * 0.44)], fill=(255, 255, 255, 120))
    d.polygon([(s * 0.30, s * 0.78), (s * 0.80, s * 0.52), (s * 0.86, s * 0.63),
               (s * 0.36, s * 0.88)], fill=(255, 255, 255, 80))
    d.rounded_rectangle([s * 0.10, s * 0.10, s * 0.90, s * 0.90], int(r * 0.7),
                        outline=(190, 232, 246, 70), width=max(1, int(s * 0.01)))
    return img


@lru_cache(maxsize=4)
def peach_img(s):
    img = Image.new("RGBA", (s * 2, s * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    c = s
    r = s * 0.86
    d.ellipse([c - r, c - r * 0.96, c + r, c + r * 0.96], fill=(255, 152, 112, 255))
    d.ellipse([c - r * 0.98, c - r * 0.98, c + r * 0.86, c + r * 0.90],
              fill=(255, 178, 132, 255))
    d.ellipse([c - r * 0.55, c - r * 0.58, c + r * 0.62, c + r * 0.30],
              fill=(255, 214, 168, 110))
    d.arc([c - r * 1.02, c - r * 0.9, c + r * 1.02, c + r * 1.02], 250, 290,
          fill=(214, 74, 62, 190), width=max(3, int(s * 0.07)))
    d.ellipse([c - r * 0.30, c - r * 0.32, c + r * 0.55, c + r * 0.40],
              fill=(255, 190, 160, 120))
    # leaf
    d.polygon([(c + r * 0.10, c - r * 0.92), (c + r * 0.62, c - r * 1.22),
               (c + r * 0.30, c - r * 0.72)], fill=(76, 168, 92, 255))
    return img


@lru_cache(maxsize=4)
def lemon_img(s):
    img = Image.new("RGBA", (s * 2, s * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    c = s
    r = s * 0.88
    d.ellipse([c - r, c - r, c + r, c + r], fill=(255, 232, 120, 255))
    d.ellipse([c - r * 0.72, c - r * 0.72, c + r * 0.72, c + r * 0.72],
              fill=(255, 250, 200, 255))
    for k in range(8):
        a = k * math.pi / 4
        d.polygon([(c, c),
                   (c + r * 0.70 * math.cos(a - 0.24), c + r * 0.70 * math.sin(a - 0.24)),
                   (c + r * 0.70 * math.cos(a + 0.24), c + r * 0.70 * math.sin(a + 0.24))],
                  fill=(255, 214, 70, 255))
    d.ellipse([c - r * 0.10, c - r * 0.10, c + r * 0.10, c + r * 0.10],
              fill=(255, 246, 176, 255))
    return img


# ---------------------------------------------------------------- scene helpers
def new_frame(bg=DEEP):
    return Image.new("RGBA", (W, H), bg + (255,))


def bg_navy(t=0.0, top=NAVY, bot=DEEP):
    f = new_frame()
    f.paste(gradimg(W, H, top, bot), (0, 0))
    return f


def glow(base, cx, cy, radius, color, alpha=1.0, power=1.6):
    d = 700
    g = radial(d, color, (0, 0, 0), power)
    blit(base, g, cx, cy, radius * 2 / d, 0, alpha)


def sparkles(base, t, n=26, seed=3, color=WHITE, size=16, spread=(W, H), cx=W / 2, cy=H / 2):
    r = np.random.default_rng(seed)
    for i in range(n):
        ph = r.random() * 6.28
        s = 0.5 + 0.5 * math.sin(t * 3.1 + ph)
        x = cx + (r.random() - 0.5) * spread[0]
        y = cy + (r.random() - 0.5) * spread[1]
        sz = size * (0.4 + r.random() * 1.0) * (0.5 + s)
        a = 0.30 + 0.7 * s
        d = ImageDraw.Draw(base, "RGBA")
        col = color + (int(190 * a),)
        d.line([x - sz, y, x + sz, y], fill=col, width=max(1, int(sz * 0.14)))
        d.line([x, y - sz, x, y + sz], fill=col, width=max(1, int(sz * 0.14)))


def droplets(base, items, t, col=ICE, speed=1.0):
    d = ImageDraw.Draw(base, "RGBA")
    for (x0, y0, r, ph) in items:
        y = y0 + (t * speed * 220 + ph * 90) % (H * 0.9)
        d.ellipse([x0, y, x0 + r, y + r * 1.5], fill=col + (150,))
        d.ellipse([x0 + r * 0.2, y + r * 0.2, x0 + r * 0.45, y + r * 0.6],
                  fill=WHITE + (170,))


def rule(base, cx, cy, w, h, col, a=255):
    ImageDraw.Draw(base, "RGBA").rounded_rectangle(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], h / 2, fill=col + (a,))


def sound_bars(base, cx, cy, n, t, col=TEAL, bar_w=18, gap=10, maxh=150, seed=5, alpha=1.0):
    r = np.random.default_rng(seed)
    pat = r.random(n)
    d = ImageDraw.Draw(base, "RGBA")
    x0 = cx - (n * (bar_w + gap) - gap) / 2
    for i in range(n):
        v = 0.25 + 0.75 * abs(math.sin(t * 6.5 + pat[i] * 9))
        hh = maxh * v * (0.5 + pat[i] * 0.6)
        col_i = mixc(col, YELLOW, pat[i])
        d.rounded_rectangle([x0 + i * (bar_w + gap), cy - hh / 2,
                             x0 + i * (bar_w + gap) + bar_w, cy + hh / 2],
                            bar_w / 2, fill=col_i + (int(235 * alpha),))


def glass(base, cx, base_y, gh, fill_frac, ices=(), t=0.0, sway=0.0):
    """tapered glass. ices = [(fx, fy, fsz, rot)] with fractions of the glass box."""
    wt = gh * 0.62
    wb = gh * 0.46
    top = base_y - gh
    lev = base_y - gh * (0.06 + 0.88 * clamp(fill_frac))
    d = ImageDraw.Draw(base, "RGBA")
    body = [(cx - wt / 2, top), (cx + wt / 2, top), (cx + wb / 2, base_y), (cx - wb / 2, base_y)]
    d.polygon(body, fill=(255, 255, 255, 24))
    # tea fill
    if fill_frac > 0.001:
        fracv = (base_y - lev) / gh
        pts = []
        n = 40
        amp = 5 + 14 * abs(sway)
        for i in range(n + 1):
            xx = lerp(cx - lerp(wt, wb, fracv) / 2, cx + lerp(wt, wb, fracv) / 2, i / n)
            yy = lev + math.sin(i / n * 6.0 + t * 7.0) * amp
            pts.append((xx, yy))
        pts += [(cx + wb / 2, base_y), (cx - wb / 2, base_y)]
        d.polygon(pts, fill=(206, 122, 34, 235))
        d.polygon(pts, fill=(245, 186, 88, 70))
    # ice cubes (fractions of glass box)
    for (fx, fy, fsz, rot) in ices:
        blit(base, ice_img(), cx + fx * wt, top + fy * gh, fsz * gh / 420.0, rot * 34, 0.88)
    # glass outline + highlights
    d.polygon(body, outline=(255, 255, 255, 150), width=max(3, int(gh * 0.007)))
    d.ellipse([cx - wt / 2, top - wt * 0.10, cx + wt / 2, top + wt * 0.10],
              outline=(255, 255, 255, 120), width=max(2, int(gh * 0.006)))
    d.line([cx - wt * 0.40, top + gh * 0.06, cx - wb * 0.36, base_y - gh * 0.04],
           fill=(255, 255, 255, 90), width=max(2, int(gh * 0.012)))
    d.line([cx + wt * 0.42, top + gh * 0.10, cx + wb * 0.38, base_y - gh * 0.10],
           fill=(255, 255, 255, 55), width=max(2, int(gh * 0.008)))


def ray_fan(base, cx, cy, t, n=18, r=1400, col=YELLOW, alpha=0.13):
    d = ImageDraw.Draw(base, "RGBA")
    for i in range(n):
        a = i * 2 * math.pi / n + t * 0.35
        w = 0.022 + 0.016 * (i % 3)
        p = [(cx, cy),
             (cx + r * math.cos(a - w), cy + r * math.sin(a - w)),
             (cx + r * math.cos(a + w), cy + r * math.sin(a + w))]
        d.polygon(p, fill=col + (int(alpha * 255),))


def burst_ring(base, cx, cy, r, w, col=YELLOW, a=180):
    d = ImageDraw.Draw(base, "RGBA")
    d.ellipse([cx - r, cy - r * 0.55, cx + r, cy + r * 0.55],
              outline=col + (a,), width=max(2, int(w)))


def squash(base, cx, cy, rx, ry, col, a=255, outl=None):
    ImageDraw.Draw(base, "RGBA").ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                                         fill=col + (a,), outline=outl)


def cam_shake(seed, t, amp=14, freq=24):
    r = np.random.default_rng(seed)
    return (math.sin(t * freq + r.random() * 6) * amp,
            math.cos(t * freq * 1.13 + r.random() * 6) * amp * 0.7)


def posterize_bg(base, t, col=YELLOW, a=0.10):
    glow(base, W * 0.5, H * 0.5, W * 0.85, col, a, 2.6)


def watermark(base, a=150):
    txt(base, "KI KI", W - 150, 92, 44, CREAM, weight="ExtraBold", tracking=4, alpha=a / 255)


# ---------------------------------------------------------------- shots
def beat(t, beats):
    """beats = [(dur, fn, transition)] -> local t, fn, transition, index."""
    acc = 0.0
    for i, (d, fn, tr) in enumerate(beats):
        if t < acc + d or i == len(beats) - 1:
            local = clamp((t - acc) / d)
            return local, fn, tr, i, (t - acc)
        acc += d
    return 1.0, beats[-1][1], beats[-1][2], len(beats) - 1, 0.0


def trans(base, t_in, kind, prev=None):
    """apply an in-transition during the first moments of a cut."""
    if kind == "cut" or t_in > 0.34:
        return base
    if kind == "flash":
        p = clamp(t_in / 0.20)
        f = Image.new("RGBA", (W, H), (255, 255, 255, int(255 * (1 - p) ** 1.4)))
        base.alpha_composite(f)
    elif kind == "whip":
        p = ease_out(clamp(t_in / 0.26))
        off = int((1 - p) * W * 0.55)
        out = Image.new("RGBA", (W, H), NAVY + (255,))
        streak = gradimg(600, H, WHITE, mixc(NAVY, TEAL, 0.4), True)
        for dx, al in ((0, 90), (40, 60), (90, 34)):
            s = streak.resize((off + dx, H), Image.BILINEAR) if off > 8 else None
            if s is not None:
                out.alpha_composite(s.convert("RGBA"), (-dx, 0))
        tmp = base.copy()
        tmp.putalpha(255)
        for dx, al in ((-38, 60), (-16, 90), (0, 255)):
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            lay.paste(tmp, (off + dx, 0))
            lay.putalpha(Image.new("L", (W, H), al))
            out.alpha_composite(lay)
        return out
    elif kind == "zoom":
        p = ease_out(clamp(t_in / 0.30))
        s = 1.0 + 0.22 * (1 - p)
        nw, nh = int(W * s), int(H * s)
        z = base.resize((nw, nh), Image.BILINEAR).crop(
            ((nw - W) // 2, (nh - H) // 2, (nw - W) // 2 + W, (nh - H) // 2 + H))
        base = z.convert("RGBA")
    return base


# ---- shot 1: hot / thirsty / boring fridge
def s1a(u, d):
    f = bg_navy(0, (34, 18, 14), (16, 8, 8))
    r = 420 + 22 * math.sin(u * 6)
    glow(f, W * 0.78, H * 0.26, r, (255, 150, 60), 0.85, 1.9)
    squash(f, W * 0.78, H * 0.26, 250, 250, (255, 176, 84), 235)
    for k in range(9):   # heat shimmer
        y = H * 0.30 + k * 46 + math.sin(u * 5 + k) * 8
        ImageDraw.Draw(f, "RGBA").line([0, y, W, y], fill=(255, 120, 60, 26), width=5)
    txt(f, "IT'S HOT.", W / 2, H * 0.56, 250, CREAM, stroke=NAVY, sw=14, pop=u * 2.6,
        shadow=(0, 0, 0), shadow_off=(10, 12))
    txt(f, "AND THE FRIDGE IS BORING", W / 2, H * 0.76, 62, YELLOW, weight="ExtraBold",
        tracking=7, pop=clamp(u * 3 - 0.7), alpha=clamp(u * 3 - 0.5))
    return f


def s1b(u, d):
    f = bg_navy(0, (18, 26, 40), (8, 12, 20))
    txt(f, "YOU'RE THIRSTY.", W / 2, H * 0.44, 210, WHITE, stroke=TEAL_D, sw=12, pop=u * 2.8)
    # big sweat drop falling
    yy = H * 0.12 + ease_in(clamp(u * 1.15)) * H * 0.62
    d2 = ImageDraw.Draw(f, "RGBA")
    d2.ellipse([W / 2 - 62, yy, W / 2 + 62, yy + 150], fill=ICE + (200,))
    d2.polygon([(W / 2 - 46, yy + 110), (W / 2, yy - 60), (W / 2 + 46, yy + 110)],
               fill=ICE + (200,))
    d2.ellipse([W / 2 - 30, yy + 40, W / 2 - 6, yy + 74], fill=WHITE + (170,))
    txt(f, "REAL COLD. REAL FAST.", W / 2, H * 0.78, 70, TEAL, weight="ExtraBold",
        tracking=8, alpha=clamp(u * 2.4 - 0.6))
    return f


def s1c(u, d):
    f = bg_navy(0, (34, 36, 40), (16, 18, 20))
    for k in range(5):
        x = lerp(W * 0.10, W * 0.90, k / 4)
        g = (76 + k * 6, 80 + k * 4, 84 + k * 3)
        ImageDraw.Draw(f, "RGBA").rounded_rectangle(
            [x - 120, H * 0.18, x + 120, H * 0.92], 30, fill=g + (255,))
        txt(f, "SUGAR", x, H * 0.38, 44, (150, 154, 158), weight="Bold", tracking=3)
    scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(scrim, "RGBA").rectangle([0, H * 0.40, W, H * 0.66], fill=(8, 10, 14, 205))
    f.alpha_composite(scrim.filter(ImageFilter.GaussianBlur(26)))
    txt(f, "BORING.", W / 2, H * 0.52, 250, (250, 250, 252), stroke=(28, 30, 34), sw=14,
        pop=clamp(u * 3.4 - 0.12), alpha=clamp(u * 4))
    rule(f, W / 2, H * 0.635, 940 * ease_out(clamp(u * 3 - 0.4)), 6, (168, 172, 176))
    txt(f, "SUGAR WATER.", W / 2, H * 0.76, 88, (208, 212, 216), weight="ExtraBold",
        tracking=10, pop=clamp(u * 3.4 - 0.5), alpha=clamp(u * 4 - 0.5))
    return f


def s1d(u, d):
    f = new_frame((6, 10, 14))
    posterize_bg(f, u, TEAL, 0.16)
    txt(f, "SO WE FIXED IT.", W / 2, H / 2, 190, WHITE, weight="ExtraBold", tracking=14,
        pop=u * 3.0, alpha=clamp(u * 3.2))
    rule(f, W / 2, H * 0.66, 420 * ease_out(u * 1.4), 8, YELLOW)
    return f


def shot1(t):
    beats = [(1.9, s1a, "cut"), (1.8, s1b, "whip"), (2.6, s1c, "whip"),
             (1.7, s1d, "flash")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 2: reveal
def s2a(u, d):
    f = bg_navy(0, (24, 40, 52), DEEP)
    glow(f, W / 2, H * 0.62, 600 + 70 * math.sin(u * 22), TEAL, 0.26, 2.0)
    dd = ImageDraw.Draw(f, "RGBA")
    w = lerp(22, 78, ease_out(u))
    beam = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(beam, "RGBA").polygon(
        [(W / 2 - w, 0), (W / 2 + w, 0), (W / 2 + w * 2.4, H), (W / 2 - w * 2.4, H)],
        fill=(255, 255, 255, 44))
    f.alpha_composite(beam.filter(ImageFilter.GaussianBlur(22)))
    return f


def s2b(u, d):
    f = bg_navy(0, (16, 34, 46), (6, 14, 22))
    posterize_bg(f, 0, TEAL, 0.20)
    ray_fan(f, W / 2, H * 0.52, u, 20, 1500, YELLOW, 0.10)
    drop = ease_out(clamp(u * 1.75))
    cy = lerp(-460, H * 0.46, drop)
    squash(f, W / 2, cy + 380, 320 * (0.4 + 0.6 * drop), 60, (0, 0, 0), 90)
    blit(f, can_img(), W / 2, cy, 700 / 1500, 0, 1.0)
    if u > 0.56:   # landing: rings + dust kick
        p = clamp((u - 0.56) * 3.2)
        burst_ring(f, W / 2, H * 0.46 + 380, 200 + 900 * ease_out(p), 16 * (1 - p) + 3, YELLOW,
                   int(190 * (1 - p)))
        burst_ring(f, W / 2, H * 0.46 + 380, 120 + 620 * ease_out(p), 10 * (1 - p) + 2, WHITE,
                   int(150 * (1 - p)))
        rr = np.random.default_rng(77)
        dd = ImageDraw.Draw(f, "RGBA")
        for _ in range(22):   # impact dust
            a = rr.uniform(0, 6.28)
            r = rr.uniform(120, 620) * ease_out(p)
            s = rr.uniform(6, 20) * (1 - p * 0.6)
            dd.ellipse([W / 2 + math.cos(a) * r * 1.5, H * 0.46 + 380 + math.sin(a) * r * 0.4,
                        W / 2 + math.cos(a) * r * 1.5 + s, H * 0.46 + 380 + math.sin(a) * r * 0.4 + s],
                       fill=(255, 232, 176, int(190 * (1 - p))))
    sparkles(f, u, 20, 9, YELLOW, 26, (1500, 600), W / 2, H * 0.5)
    txt(f, "KI KI", W / 2, H * 0.875, 240, CREAM, stroke=NAVY, sw=14,
        pop=clamp(u * 2.2 - 0.55), tracking=6)
    txt(f, "ICED TEA, DONE RIGHT", W / 2, H * 0.975, 54, YELLOW, weight="Bold",
        tracking=12, alpha=clamp(u * 2.6 - 1.1))
    return f


def s2c(u, d):
    f = bg_navy(0, (18, 44, 52), (6, 16, 22))
    posterize_bg(f, 0, TEAL, 0.24)
    # fake y-axis turn: squash the width, never rotate the flat art in-plane
    turn = ease_io(u)
    sx = max(0.26, abs(math.cos(2 * math.pi * turn)))
    ci = can_img()
    csq = ci.resize((max(2, int(ci.size[0] * sx)), ci.size[1]), Image.BILINEAR)
    blit(f, csq, W / 2, H * 0.50, (0.44 + 0.05 * (1 - sx)) * (1300 / 1500), 0, 1.0,
         key=("can_sq", round(sx, 3)))
    shx = lerp(-400, W + 400, ease_io(u))       # shine sweep
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(lay, "RGBA").polygon(
        [(shx - 120, 0), (shx + 120, 0), (shx - 60, H), (shx - 300, H)],
        fill=(255, 255, 255, 70))
    f.alpha_composite(lay.filter(ImageFilter.GaussianBlur(26)))
    txt(f, "REAL BREW", W / 2, H * 0.93, 92, CREAM, weight="ExtraBold", tracking=16,
        alpha=clamp(u * 2.0 - 0.4))
    return f


def s2d(u, d):
    f = bg_navy(0, (12, 30, 42), (4, 10, 16))
    posterize_bg(f, 0, YELLOW, 0.14)
    blit(f, can_img(), W * 0.5, H * 0.47, 0.50 + 0.02 * math.sin(u * 4), 0, 1.0)
    for k, (img, rad, sz, ph) in enumerate([(peach_img(120), 470, 0.42, 0.0),
                                            (lemon_img(110), 540, 0.40, 3.1),
                                            (ice_img(), 430, 0.30, 1.6)]):
        a = t_orb = u * 2.4 + ph
        blit(f, img, W * 0.5 + math.cos(a) * rad, H * 0.50 + math.sin(a) * rad * 0.42,
             sz, math.degrees(a) * 0.5, 1.0)
    sparkles(f, u, 22, 4, CREAM, 22, (1300, 700), W / 2, H / 2)
    txt(f, "KI KI!", W * 0.5, H * 0.90, 120, YELLOW, stroke=NAVY, sw=10,
        pop=clamp(u * 3 - 0.3), tracking=6)
    return f


def shot2(t):
    beats = [(0.55, s2a, "flash"), (3.6, s2b, "cut"), (2.1, s2c, "whip"),
             (1.75, s2d, "zoom")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 3: pour
def s3a(u, d):
    f = bg_navy(0, (10, 62, 66), (5, 24, 30))
    posterize_bg(f, 0, TEAL, 0.30)
    ices = [(-0.16, 0.62, 0.44, 0.0), (0.14, 0.72, 0.40, 0.1)]
    glass(f, W / 2, H * 0.86, 620, 0.05, ices, u)
    txt(f, "REAL ICE FIRST.", W / 2, H * 0.15, 76, CREAM, weight="ExtraBold", tracking=10,
        alpha=clamp(u * 2.5 - 0.2))
    return f


def s3b(u, d):
    f = bg_navy(0, (10, 58, 62), (5, 22, 28))
    posterize_bg(f, 0, TEAL, 0.26)
    gy = H * 0.98
    gh = 720
    fill = ease_out(clamp(u * 0.92))
    lev = gy - gh * (0.06 + 0.88 * fill)
    # pour stream from above, stopping exactly at the rising surface
    sx = W / 2 + math.sin(u * 9) * 10
    sw = 46 + 10 * math.sin(u * 13)
    dd = ImageDraw.Draw(f, "RGBA")
    dd.polygon([(sx - sw / 2, -10), (sx + sw / 2, -10),
                (sx + sw / 2 - 8, lev + 26), (sx - sw / 2 - 8, lev + 26)],
               fill=(226, 148, 52, 240))
    dd.polygon([(sx - sw / 6, -10), (sx + sw / 6, -10),
                (sx + sw / 6 - 6, lev + 26), (sx - sw / 6 - 6, lev + 26)],
               fill=(255, 210, 130, 160))
    glass(f, W / 2, gy, gh, fill,
          [(-0.14, (lev - (gy - gh)) / gh - 0.14, 0.30, 0.2),
           (0.11, (lev - (gy - gh)) / gh - 0.08, 0.28, 0.0),
           (0.02, (lev - (gy - gh)) / gh - 0.30, 0.24, 0.4)],
          u, sway=0.6)
    # splash crown on the surface
    rr = np.random.default_rng(21)
    for _ in range(16):
        a = rr.uniform(0, 6.28)
        r = rr.uniform(30, 190) * (0.4 + 0.6 * fill)
        x = W / 2 + math.cos(a) * r
        y = lev - abs(math.sin(a)) * rr.uniform(20, 130)
        s = rr.uniform(8, 22)
        dd.ellipse([x, y, x + s, y + s * 1.5], fill=(255, 208, 130, 200))
    burst_ring(f, W / 2, lev, 240, 6, (255, 236, 190), 120)
    txt(f, "POURED OVER REAL ICE", W / 2, H * 0.13, 66, CREAM, weight="ExtraBold",
        tracking=9, alpha=clamp(u * 2.2 - 0.1))
    return f


def s3c(u, d):
    f = bg_navy(0, (12, 52, 58), (5, 20, 26))
    posterize_bg(f, 0, TEAL, 0.22)
    gy = H * 0.98
    glass(f, W / 2, gy, 720, 0.97,
          [(-0.13, 0.15, 0.29, 0.1), (0.12, 0.25, 0.27, 0.6), (0.01, 0.07, 0.25, 0.3)],
          u, sway=0.25)
    rr = np.random.default_rng(31)
    items = [(W / 2 - 300 + rr.uniform(0, 600), rr.uniform(0, 400), rr.uniform(8, 18),
              rr.random()) for _ in range(14)]
    droplets(f, items, u, ICE, 1.0)
    txt(f, "NO SHORTCUTS. NO POWDER.", W / 2, H * 0.14, 78, CREAM, weight="ExtraBold",
        tracking=9, pop=clamp(u * 2.6 - 0.2))
    watermark(f, 120)
    return f


def shot3(t):
    beats = [(1.25, s3a, "whip"), (4.9, s3b, "cut"), (1.85, s3c, "zoom")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 4: ice macro / crackle
def s4a(u, d):
    f = new_frame((150, 88, 26))
    f.paste(gradimg(W, H, (206, 132, 44), (140, 74, 18)), (0, 0))
    for k in range(3):
        blit(f, ice_img(), W * (0.22 + 0.28 * k) + math.sin(u * 3 + k) * 14,
             H * (0.34 + 0.16 * k), 0.9 + 0.12 * math.sin(u * 2 + k), 8 * k, 1.0)
    # crack lines growing
    rr = np.random.default_rng(41)
    dd = ImageDraw.Draw(f, "RGBA")
    for s in range(5):
        x0 = rr.uniform(W * 0.15, W * 0.85)
        y0 = rr.uniform(H * 0.15, H * 0.85)
        pts = [(x0, y0)]
        a = rr.uniform(0, 6.28)
        seg = int(14 * clamp(u * 1.2))
        for i in range(seg):
            a += rr.uniform(-0.6, 0.6)
            x0 += math.cos(a) * 34
            y0 += math.sin(a) * 34
            pts.append((x0, y0))
        if len(pts) > 1:
            dd.line(pts, fill=(255, 255, 255, 210), width=max(1, int(2 + 4 * (1 - u))))
    sound_bars(f, W / 2, H * 0.80, 22, u, TEAL, 20, 12, 130, 7, 0.9)
    txt(f, "CRACKLE.", W / 2, H * 0.16, 130, WHITE, stroke=(120, 60, 10), sw=10,
        pop=clamp(u * 2.6 - 0.1), tracking=6)
    return f


def s4b(u, d):
    f = new_frame((176, 104, 30))
    f.paste(gradimg(W, H, (226, 152, 56), (150, 82, 22)), (0, 0))
    blit(f, ice_img(), W * 0.5, H * 0.52, 1.25, 12 * math.sin(u * 8), 1.0)
    rr = np.random.default_rng(51)
    dd = ImageDraw.Draw(f, "RGBA")
    for k in range(70):     # bubbles rushing
        if k % 3:
            continue
        x = rr.uniform(W * 0.05, W * 0.95)
        y = H - ((u * rr.uniform(700, 1500) + rr.uniform(0, H)) % H)
        s = rr.uniform(10, 42)
        dd.ellipse([x, y, x + s, y + s], outline=(255, 255, 255, 150), width=3)
        dd.ellipse([x + s * 0.2, y + s * 0.2, x + s * 0.45, y + s * 0.45],
                   fill=(255, 255, 255, 110))
    ox, oy = cam_shake(3, u, 16, 26)
    txt(f, "THAT'S THE POINT.", W / 2 + ox, H * 0.30 + oy, 150, CREAM, stroke=(110, 54, 8),
        sw=12, pop=clamp(u * 3 - 0.1), tracking=4)
    return f


def s4c(u, d):
    f = new_frame((160, 94, 28))
    f.paste(gradimg(W, H, (216, 140, 48), (128, 66, 16)), (0, 0))
    blit(f, ice_img(), W * 0.36, H * 0.52, 0.9, 14, 1.0)
    blit(f, ice_img(), W * 0.62, H * 0.60, 0.8, -18, 1.0)
    sparkles(f, u, 18, 6, WHITE, 30, (1400, 700), W / 2, H / 2)
    txt(f, "ICE COLD.", W / 2, H * 0.22, 140, WHITE, stroke=(120, 60, 10), sw=10,
        pop=clamp(u * 3 - 0.2), tracking=8)
    watermark(f, 130)
    return f


def shot4(t):
    beats = [(3.9, s4a, "cut"), (2.1, s4b, "flash"), (2.0, s4c, "whip")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 5: flavour
def s5a(u, d):
    f = bg_navy(0, (46, 20, 22), (18, 8, 12))
    posterize_bg(f, 0, PEACH, 0.22)
    for k, (img, ph, sz) in enumerate([(peach_img(200), 0.0, 0.55), (lemon_img(180), 1.4, 0.5)]):
        p = clamp(u * 1.5 - ph * 0.5)
        x = lerp(-500, W * (0.32 + 0.36 * k), ease_out(p))
        y = H * (0.46 + 0.06 * math.sin(u * 6 + ph))
        blit(f, img, x, y, sz, 400 * ease_out(p), 1.0)
        for j in range(9):   # juice droplets
            dd = ImageDraw.Draw(f, "RGBA")
            a = j * 0.7 + u * 5
            x2 = x + math.cos(a) * 260 * p
            y2 = y + math.sin(a) * 200 * p
            dd.ellipse([x2, y2, x2 + 18, y2 + 24],
                       fill=(mixc(PEACH, YELLOW, j / 9)) + (200,))
    txt(f, "PEACH.", W * 0.30, H * 0.80, 130, PEACH_L, stroke=NAVY, sw=10,
        pop=clamp(u * 3 - 0.2), tracking=6)
    txt(f, "LEMON.", W * 0.71, H * 0.86, 130, (255, 234, 130), stroke=NAVY, sw=10,
        pop=clamp(u * 3 - 0.8), tracking=6)
    return f


def s5b(u, d):
    f = bg_navy(0, (16, 34, 46), (5, 12, 20))
    posterize_bg(f, 0, YELLOW, 0.16)
    blit(f, can_img(), W / 2, H * 0.50, 0.56, 0, 1.0)
    d2 = ImageDraw.Draw(f, "RGBA")
    for k in range(3):
        r = 380 + k * 90 + math.sin(u * 5 + k) * 26
        d2.ellipse([W / 2 - r, H * 0.52 - r * 0.34, W / 2 + r, H * 0.52 + r * 0.34],
                   outline=mixc(TEAL, YELLOW, k / 3) + (150 - k * 34,),
                   width=max(2, 8 - k * 2))
    sparkles(f, u, 26, 8, CREAM, 24, (1300, 700), W / 2, H / 2)
    txt(f, "A LITTLE SPARKLE", W / 2, H * 0.94, 84, CREAM, weight="ExtraBold", tracking=14,
        pop=clamp(u * 2.4 - 0.3))
    return f


def s5c(u, d):
    f = bg_navy(0, (10, 40, 46), (4, 14, 20))
    posterize_bg(f, 0, TEAL, 0.24)
    labels = [("PEACH", peach_img(120), PEACH), ("LEMON", lemon_img(110), YELLOW),
              ("REAL ICE", ice_img(), ICE)]
    for k, (name, img, col) in enumerate(labels):
        cx = W * (0.22 + 0.28 * k)
        p = clamp(u * 2.6 - k * 0.22)
        r = 130 * ease_back(p)
        ImageDraw.Draw(f, "RGBA").ellipse([cx - r, H * 0.46 - r, cx + r, H * 0.46 + r],
                                          outline=col + (230,), width=6)
        blit(f, img, cx, H * 0.46, 1.05 * p, 12 * math.sin(u * 4 + k), 1.0)
        txt(f, name, cx, H * 0.60, 58, col, weight="ExtraBold", tracking=6, alpha=clamp(u * 3 - k * 0.2))
    txt(f, "KI KI!", W / 2, H * 0.14, 96, YELLOW, stroke=NAVY, sw=8,
        alpha=clamp(u * 3 - 0.3), tracking=6)
    watermark(f, 130)
    return f


def shot5(t):
    beats = [(3.05, s5a, "whip"), (3.0, s5b, "cut"), (1.95, s5c, "flash")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 6: vibe / skyline
@lru_cache(maxsize=2)
def skyline(seed=2):
    r = np.random.default_rng(seed)
    b = []
    x = 0
    while x < 2600:
        w = r.uniform(70, 190)
        h = r.uniform(120, 420)
        b.append((x, w, h))
        x += w + r.uniform(8, 36)
    return b


def draw_skyline(f, off, ybase, col, a=255, scale=1.0):
    dd = ImageDraw.Draw(f, "RGBA")
    for (x, w, h) in skyline():
        x2 = x - off
        if -300 < x2 < W + 300:
            dd.rectangle([x2, ybase - h * scale, x2 + w, ybase], fill=col + (a,))


def s6a(u, d):
    f = new_frame()
    f.paste(gradimg(W, H, (255, 152, 96), (26, 26, 68)), (0, 0))
    f.paste(gradimg(W, int(H * 0.55), (255, 194, 120), (255, 238, 190)), (0, 0))
    glow(f, W * 0.62, H * 0.52, 380, (255, 214, 130), 0.95, 1.4)
    squash(f, W * 0.62, H * 0.52, 190, 190, (255, 246, 214), 255)
    draw_skyline(f, u * 60, H * 0.74, (58, 40, 74), 210, 0.75)
    draw_skyline(f, u * 220, H * 0.86, (26, 18, 40), 240, 1.0)
    dd = ImageDraw.Draw(f, "RGBA")
    for k in range(7):     # birds
        bx = W * 0.15 + k * 120 + u * 60
        by = H * 0.28 + math.sin(u * 3 + k) * 16 + k * 8
        dd.arc([bx, by, bx + 34, by + 22], 200, 340, fill=(40, 26, 44, 210), width=3)
        dd.arc([bx + 30, by, bx + 64, by + 22], 200, 340, fill=(40, 26, 44, 210), width=3)
    blit(f, can_img(), W * 0.22, H * 0.72, 0.44, -6, 1.0)
    txt(f, "ROOFTOPS.", W * 0.68, H * 0.18, 120, CREAM, stroke=(70, 30, 40), sw=10,
        pop=clamp(u * 2.6 - 0.1), tracking=8)
    return f


def s6b(u, d):
    f = new_frame()
    f.paste(gradimg(W, H, (22, 40, 62), (10, 18, 30)), (0, 0))
    posterize_bg(f, 0, TEAL, 0.20)
    dd = ImageDraw.Draw(f, "RGBA")
    # perspective road
    dd.polygon([(W * 0.36, H * 0.30), (W * 0.64, H * 0.30), (W * 1.06, H), (-W * 0.06, H)],
               fill=(38, 42, 50, 255))
    for k in range(9):
        p = ((u * 1.6 + k / 9) % 1.0)
        y = lerp(H * 0.30, H, p ** 1.8)
        w = lerp(10, 90, p)
        h = lerp(8, 46, p)
        dd.rounded_rectangle([W / 2 - w / 2, y, W / 2 + w / 2, y + h], w / 6,
                            fill=(255, 214, 120, 210))
    dd.rectangle([0, H * 0.28, W, H * 0.31], fill=(90, 200, 190, 90))
    blit(f, can_img(), W * 0.80, H * 0.62, 0.34, 10, 1.0)
    txt(f, "ROAD TRIPS.", W * 0.32, H * 0.52, 118, CREAM, stroke=NAVY, sw=10,
        pop=clamp(u * 2.8 - 0.15), tracking=8)
    return f


def s6c(u, d):
    f = bg_navy(0, (26, 30, 56), (10, 12, 26))
    posterize_bg(f, 0, PEACH, 0.20)
    cx, cy, r = W * 0.27, H * 0.50, 205
    dd = ImageDraw.Draw(f, "RGBA")
    dd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 240, 210, 40),
               outline=CREAM + (170,), width=6)
    for k in range(12):   # hour marks
        a = k * math.pi / 6
        dd.line([cx + math.cos(a) * r * 0.86, cy + math.sin(a) * r * 0.86,
                 cx + math.cos(a) * r * 0.95, cy + math.sin(a) * r * 0.95],
                fill=CREAM + (150,), width=4)
    spin = u * 16
    for mult, ln, wd, col in ((1.0, r * 0.55, 10, CREAM), (12, r * 0.78, 6, YELLOW)):
        a = -spin * mult
        dd.line([cx, cy, cx + math.sin(a) * ln, cy - math.cos(a) * ln],
                fill=col + (240,), width=wd)
    dd.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=PEACH + (255,))
    txt(f, "LONG AFTERNOONS", W * 0.70, H * 0.47, 96, CREAM, weight="ExtraBold",
        tracking=5, pop=clamp(u * 2.6 - 0.1))
    txt(f, "...THAT NEVER END", W * 0.70, H * 0.585, 58, YELLOW, weight="Bold",
        tracking=9, alpha=clamp(u * 2.6 - 0.8))
    watermark(f, 120)
    return f


def shot6(t):
    beats = [(4.0, s6a, "whip"), (2.05, s6b, "cut"), (1.95, s6c, "whip")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 7: benefit cards
def card_base(col):
    f = new_frame()
    f.paste(gradimg(W, H, mixc(col, WHITE, 0.10), mixc(col, DEEP, 0.35)), (0, 0))
    posterize_bg(f, 0, col, 0.16)
    return f


def s7a(u, d):
    f = card_base(TEAL_D)
    txt(f, "LESS SUGAR", W / 2, H * 0.30, 168, CREAM, stroke=DEEP, sw=10,
        pop=clamp(u * 3 - 0.1), tracking=6)
    w = lerp(1200, 420, ease_out(u))
    rule(f, W / 2, H * 0.52, 1300, 60, WHITE, 60)
    rule(f, W / 2, H * 0.52, w, 60, YELLOW, 255)
    for k in range(9):     # sugar cubes vanishing
        if k / 9 < 1 - u:
            blit(f, ice_img(), W * (0.16 + 0.085 * k), H * 0.70, 0.22, 20 * k, 0.35)
    txt(f, "-60% SUGAR", W / 2, H * 0.86, 86, YELLOW, weight="ExtraBold", tracking=10,
        alpha=clamp(u * 2.6 - 0.5))
    return f


def s7b(u, d):
    f = card_base((150, 60, 60))
    txt(f, "NO FAKE SYRUP", W / 2, H * 0.30, 150, CREAM, stroke=DEEP, sw=10,
        pop=clamp(u * 3 - 0.1), tracking=5)
    # syrup bottle + cross
    cx, cy = W / 2, H * 0.62
    dd = ImageDraw.Draw(f, "RGBA")
    dd.rounded_rectangle([cx - 90, cy - 150, cx + 90, cy + 150], 24, fill=(120, 86, 60, 235))
    dd.rounded_rectangle([cx - 40, cy - 210, cx + 40, cy - 150], 12, fill=(90, 62, 44, 235))
    dd.rounded_rectangle([cx - 60, cy + 40, cx + 60, cy + 150], 16, fill=(160, 118, 84, 200))
    p = ease_out(clamp(u * 1.5))
    for sgn in (1, -1):
        dd.line([cx - 200 * p * sgn, cy - 200 * p, cx + 200 * p * sgn, cy + 200 * p],
                fill=(255, 82, 82, 245), width=int(22))
    txt(f, "ZERO SYRUP", W / 2, H * 0.90, 78, (255, 190, 190), weight="ExtraBold",
        tracking=10, alpha=clamp(u * 2.6 - 0.5))
    return f


def s7c(u, d):
    f = card_base((30, 110, 60))
    txt(f, "REAL BREW", W / 2, H * 0.28, 170, CREAM, stroke=DEEP, sw=10,
        pop=clamp(u * 3 - 0.1), tracking=6)
    rr = np.random.default_rng(61)
    leaf = Image.new("RGBA", (132, 76), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf, "RGBA")
    ld.ellipse([6, 8, 126, 68], fill=(58, 152, 84, 245))
    ld.ellipse([22, 20, 110, 56], fill=(92, 196, 116, 210))
    ld.line([16, 38, 116, 38], fill=(228, 255, 234, 220), width=4)
    for k in range(12):   # tea leaves drifting
        x = W * 0.16 + rr.uniform(0, W * 0.68)
        y = H * 0.42 + rr.uniform(0, H * 0.34) + math.sin(u * 4 + k) * 14
        blit(f, leaf, x, y, 1.0 + 0.08 * math.sin(u * 5 + k), rr.uniform(-55, 55), 1.0)
    p = clamp(u * 2.2 - 0.2)
    dd = ImageDraw.Draw(f, "RGBA")
    dd.ellipse([W / 2 - 110, H * 0.80 - 110, W / 2 + 110, H * 0.80 + 110],
               outline=TEAL + (240,), width=10)
    dd.line([W / 2 - 54, H * 0.80, W / 2 - 12, H * 0.80 + 46], fill=TEAL + (250,),
            width=int(16 * p))
    dd.line([W / 2 - 12, H * 0.80 + 46, W / 2 + 60, H * 0.80 - 52], fill=TEAL + (250,),
            width=int(16 * p))
    txt(f, "BREWED, NOT MIXED", W / 2, H * 0.08, 56, (200, 255, 214), weight="Bold",
        tracking=10, alpha=clamp(u * 2.4 - 0.3))
    return f


def s7d(u, d):
    f = card_base((140, 96, 20))
    txt(f, "ALL FLAVOR", W / 2, H * 0.26, 168, CREAM, stroke=DEEP, sw=10,
        pop=clamp(u * 3 - 0.1), tracking=6)
    for k, (lab, col) in enumerate([("PEACH", PEACH), ("LEMON", YELLOW), ("TEA", AMBER_L)]):
        hgt = 300 * ease_out(clamp(u * 1.4 - k * 0.08))
        x = W * (0.22 + 0.28 * k) - 80
        ImageDraw.Draw(f, "RGBA").rounded_rectangle(
            [x, H * 0.82 - hgt, x + 160, H * 0.82], 16, fill=col + (245,))
        txt(f, lab, x + 80, H * 0.86, 44, CREAM, weight="Bold", tracking=4)
    blit(f, can_img(), W * 0.86, H * 0.60, 0.34, 0, 1.0)
    return f


def shot7(t):
    beats = [(2.0, s7a, "cut"), (2.0, s7b, "whip"), (2.0, s7c, "whip"),
             (2.0, s7d, "whip")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


# ---- shot 8: CTA
def s8a(u, d):
    f = bg_navy(0, (14, 32, 46), (4, 10, 18))
    posterize_bg(f, 0, TEAL, 0.18)
    ray_fan(f, W * 0.5, H * 0.50, u * 1.4, 16, 1400, TEAL, 0.07)
    blit(f, can_img(), W * 0.5, H * 0.50, lerp(0.46, 0.56, ease_out(u)), 0, 1.0)
    sparkles(f, u, 20, 12, YELLOW, 24, (1500, 800), W / 2, H / 2)
    txt(f, "KI KI", W * 0.76, H * 0.42, 180, CREAM, stroke=NAVY, sw=12,
        pop=clamp(u * 2.6 - 0.1), tracking=4, rot=-4)
    txt(f, "ICED TEA", W * 0.76, H * 0.54, 74, YELLOW, weight="ExtraBold", tracking=12,
        alpha=clamp(u * 2.6 - 0.5), rot=-4)
    txt(f, "KEEP IT COLD. KEEP IT CLOSE.", W / 2, H * 0.95, 68, CREAM, weight="Bold",
        tracking=10, alpha=clamp(u * 2.4 - 0.0))
    return f


def s8b(u, d):
    f = bg_navy(0, (20, 48, 56), (6, 16, 24))
    posterize_bg(f, 0, TEAL, 0.24)
    blit(f, can_img(), W * 0.34 - 30 * ease_out(u), H * 0.50, 0.50, -8, 1.0)
    blit(f, can_img(), W * 0.66 + 30 * ease_out(u), H * 0.50, 0.50, 8, 1.0)
    if u > 0.42:
        p = clamp((u - 0.42) * 4)
        burst_ring(f, W / 2, H * 0.50, 200 + 700 * ease_out(p), 14 * (1 - p) + 3, WHITE,
                   int(200 * (1 - p)))
        sparkles(f, u, 18, 14, CREAM, 26, (700, 500), W / 2, H * 0.50)
    txt(f, "SHARE ONE.", W / 2, H * 0.90, 96, CREAM, weight="ExtraBold", tracking=10,
        pop=clamp(u * 2.6 - 0.3))
    watermark(f, 140)
    return f


def s8c(u, d):
    f = bg_navy(0, (10, 24, 36), (3, 8, 14))
    posterize_bg(f, 0, YELLOW, 0.14)
    a = clamp(u * 1.6)
    txt(f, "KI KI", W / 2, H * 0.40, 330, CREAM, stroke=NAVY, sw=16, tracking=8,
        pop=a, alpha=clamp(a * 1.6))
    rule(f, W / 2, H * 0.55, 1000 * ease_out(clamp(u * 1.8 - 0.2)), 8, YELLOW)
    txt(f, "ICED TEA", W / 2, H * 0.63, 96, YELLOW, weight="ExtraBold", tracking=22,
        alpha=clamp(u * 2 - 0.5))
    txt(f, "PEACH + LEMON  |  REAL ICE", W / 2, H * 0.74, 54, CREAM, weight="Bold",
        tracking=8, alpha=clamp(u * 2 - 0.9))
    blit(f, can_img(), W * 0.5, H * 0.92, 0.22, 0, clamp(u * 2 - 1.2))
    sparkles(f, u, 26, 16, CREAM, 26, (1500, 800), W / 2, H / 2)
    if u > 0.72:
        p = clamp((u - 0.72) / 0.28)
        f.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(200 * p))))
    return f


def shot8(t):
    beats = [(3.9, s8a, "flash"), (2.05, s8b, "cut"), (2.05, s8c, "zoom")]
    u, fn, tr, i, ti = beat(t, beats)
    return trans(fn(u, 0), ti, tr)


SHOTS = [shot1, shot2, shot3, shot4, shot5, shot6, shot7, shot8]


def finish(f, seed=0):
    f.alpha_composite(vignette())
    f.alpha_composite(grain(seed % 6))
    return f.convert("RGB")


def render_frame(t):
    k = min(int(t // SHOT), NSHOTS - 1)
    f = SHOTS[k](t - k * SHOT)
    return finish(f, int(t * FPS))


# ---------------------------------------------------------------- cut table (authoritative pacing)
CUTS = [
    (0.0, "IT'S HOT."), (1.9, "YOU'RE THIRSTY."), (3.7, "BORING."), (6.3, "SO WE FIXED IT."),
    (8.0, "can reveal"), (8.55, "wordmark"), (12.15, "spin + shine"), (14.25, "orbit"),
    (16.0, "glass + ice"), (17.25, "pour"), (22.15, "full glass"),
    (24.0, "macro crackle"), (27.9, "bubbles + shake"), (30.0, "ice cold"),
    (32.0, "peach/lemon"), (35.05, "orbit sparkle"), (38.05, "icon trio"),
    (40.0, "rooftops"), (44.0, "road trip"), (46.05, "clock"),
    (48.0, "less sugar"), (50.0, "no syrup"), (52.0, "real brew"), (54.0, "all flavor"),
    (56.0, "end card"), (59.95, "can clink"), (62.0, "logo lockup"),
]


def cut_samples():
    """sample each cut mid-beat (past its transition) for honest review frames."""
    out = []
    for i, (t, label) in enumerate(CUTS):
        nxt = CUTS[i + 1][0] if i + 1 < len(CUTS) else TOTAL
        L = nxt - t
        out.append((t + min(max(L * 0.45, 0.35), L - 0.05), t, label))
    return out


def render_stills():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    tags = ["01_hot", "02_thirsty", "03_boring", "04_fixed",
            "05_reveal", "06_wordmark", "07_spin", "08_orbit",
            "09_glass", "10_pour", "11_full", "12_crackle", "13_shake", "14_icecold",
            "15_flavour", "16_sparkle", "17_icons", "18_rooftops", "19_road", "20_clock",
            "21_sugar", "22_syrup", "23_brew", "24_flavor", "25_endcard", "26_clink", "27_lockup"]
    mats = []
    for i, (s, t, label) in enumerate(cut_samples()):
        fr = render_frame(s)
        p = f"{FRAMES_DIR}/{i+1:02d}_{tags[i]}.png"
        fr.resize((960, 540), Image.LANCZOS).save(p)
        small = fr.resize((640, 360), Image.LANCZOS).convert("RGBA")
        cap = text_img(f"{i+1:02d}  {t:5.1f}s  {label}", INT, 22, "Bold", (255, 255, 255),
                       (0, 0, 0), 3, 0)
        blit(small, cap, 12 + cap.size[0] / 2, 344, 1.0, 0, 0.95)
        mats.append(small)
        print(f"{i+1:02d}  beat {t:5.1f}s  sample {s:5.2f}s  {label}")
    cols = 4
    rows = (len(mats) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 640, rows * 360), (0, 0, 0))
    for i, im in enumerate(mats):
        sheet.paste(im.convert("RGB"), ((i % cols) * 640, (i // cols) * 360))
    sheet.save(f"{OUTDIR}/contact_sheet.jpg", quality=90)
    print("wrote", f"{OUTDIR}/contact_sheet.jpg")


def render_range(f0, f1, out):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
           "-video_track_timescale", "30000", "-g", "60", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(render_frame(i / FPS).tobytes())
    p.stdin.close()
    p.wait()
    print(f"chunk {f0}-{f1} -> {out}", flush=True)


def render_video():
    n = int(TOTAL * FPS)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", f"{OUTDIR}/video_silent.mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        f = render_frame(i / FPS)
        p.stdin.write(f.tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{n} ({i/n*100:.0f}%)", flush=True)
    p.stdin.close()
    p.wait()
    print("wrote", f"{OUTDIR}/video_silent.mp4")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stills"
    if mode == "stills":
        render_stills()
    elif mode == "range":
        render_range(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        render_video()
