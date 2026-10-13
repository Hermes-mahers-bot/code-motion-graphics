"""Validate this skill's copy-paste kit: pulls the code blocks straight out of SKILL.md,
execs them, renders a frame, and checks beat() boundary dispatch.

Run after editing SKILL.md so the documented kit can never rot into something that does not work:
    python3 validate_kit.py
Exit 0 = the documented code is real, runnable code.
"""
import re
import sys
import textwrap

from PIL import Image

SK = __file__.replace("scripts/validate_kit.py", "SKILL.md")
src = open(SK).read()
blocks = [textwrap.dedent(b) for b in re.findall(r"```python\n(.*?)```", src, re.S)]

fails = []
kit = [b for b in blocks if "def text_img" in b]
beat_blk = [b.split("\nBEATS =")[0] for b in blocks if "def beat(" in b]
chroma_blk = [b for b in blocks if "def chroma" in b]
if len(kit) != 1 or len(beat_blk) != 1:
    sys.exit(f"block discovery failed: kit={len(kit)} beat={len(beat_blk)}")

ns = {}
try:
    exec(kit[0], ns)
    W, H = ns["W"], ns["H"]
    f = Image.new("RGBA", (W, H), (11, 26, 38, 255))
    f.paste(ns["gradimg"](W, H, (16, 34, 46), (5, 13, 20)), (0, 0))
    ns["glow"](f, W * 0.75, H * 0.28, 420, (255, 150, 60), 0.85, 1.9)
    ns["txt"](f, "IT'S HOT.", W / 2, H * 0.52, 250, (255, 247, 230),
              stroke=(11, 26, 38), sw=14, tracking=3, pop=0.8,
              shadow=(0, 0, 0), shadow_off=(10, 12))
    ns["blit"](f, ns["text_img"]("ICED TEA", ns["INT"], 54, "Bold", (255, 201, 60), None, 0, 12),
               W / 2, H * 0.78, 1.0)
    ns["blit"](f, ns["radial"](600, (206, 236, 246), (0, 0, 0), 0.6), W * 0.2, H * 0.7,
               260 / 600, 0, 0.9, key=("ice", 0))
    out = ns["finish"](f, 3)
    out.save("/tmp/skill_kit_test.png")
    print("kit primitives OK ->", out.size, out.mode)
except Exception as e:
    fails.append(f"kit: {type(e).__name__}: {e}")

try:
    ns2 = {}
    exec("def clamp(x, a=0.0, b=1.0):\n    return max(a, min(b, x))\n" + beat_blk[0], ns2)
    bea = [(3.9, "a", "cut"), (2.1, "b", "flash"), (2.0, "c", "whip")]
    for t, want in [(0.0, "a"), (3.89, "a"), (3.9, "b"), (5.99, "b"), (6.0, "c"), (7.99, "c")]:
        got = ns2["beat"](t, bea)[1]
        if got != want:
            fails.append(f"beat({t}) -> {got}, expected {want}")
    print("beat() boundary dispatch OK")
except Exception as e:
    fails.append(f"beat: {type(e).__name__}: {e}")

if chroma_blk:
    try:
        ns3 = {}
        exec("import numpy as np\nfrom PIL import Image\n" + chroma_blk[0], ns3)
        ns3["chroma"](Image.new("RGB", (64, 64), (100, 120, 140)), 4)
        print("chroma() OK")
    except Exception as e:
        fails.append(f"chroma: {type(e).__name__}: {e}")

print("FAILURES:", fails if fails else "none - documented kit verified")
sys.exit(1 if fails else 0)
