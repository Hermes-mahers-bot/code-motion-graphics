# AGENTS.md - runbook for AI agents working in this repo

## Objective

Generate short animated ads (30-90 s) entirely from code: PIL/numpy frames, a numpy-synthesised
score, edge-tts voiceover, ffmpeg assembly. The deliverable is a playable mp4 plus a stills
storyboard, verified with numbers, not vibes.

## Read first

1. `SKILL.md` - the full playbook. The timing model (section 1), the renderer kit (3), the cool
   factor cookbook (4), audio (5), verification gates (7) and the pitfall list (8) are all
   load-bearing. Do not write renderer code before reading it.
2. `example/kiki-iced-tea/render.py` - the working reference. The kit in SKILL.md section 3 is
   lifted from it, so it is the ground truth when the two ever disagree.
3. `scripts/validate_kit.py` - proves the code printed in SKILL.md actually runs.

## Environment

- python3 with Pillow and numpy, ffmpeg (libx264 + aac), edge-tts, and three font files:
  `~/.fonts/Poppins-ExtraBold.ttf`, `~/.fonts/Inter.ttf`, `~/.fonts/JetBrainsMono.ttf`.
  Font paths are overridable through `POP_FONT` / `INT_FONT` / `MONO_FONT`.
- No GPU, no network except the edge-tts voice stage.
- Every path in the example scripts is derived from `__file__`, so the directory can be copied
  anywhere. Run the scripts from inside the example directory.

## Commands, with expected cost

```bash
python3 vo.py             # ~30 s   -> vo/*.wav, vo/timing.json (11 lines)
python3 score.py          # ~6 s    -> mix/audio.wav (64.0 s)
python3 render.py stills  # ~20 s   -> contact_sheet.jpg + stills/*.png  (DO THIS BEFORE RENDERING)
bash render_all.sh        # ~5 min  -> chunks/chunk_*.mp4, video_silent.mp4 (1920 frames)
bash mix_and_mux.sh       # ~2 min  -> kiki_ad.mp4, kiki_ad_deliver.mp4, kiki_ad_720p.mp4, poster.jpg
python3 verify.py         # ~30 s   -> VO audibility table, check/final_*.png
python3 ../scripts/validate_kit.py   # ~3 s, exit 0 = documented kit still works
```

## Definition of done (all of these, with evidence)

- `ffprobe` on the final mp4: `nb_frames == TOTAL * FPS` exactly (1920 for 64 s at 30 fps),
  `duration` exact, `avg_frame_rate` 30/1, video 1920x1080 h264 plus an aac stream.
- `volumedetect` on the FINAL mp4 (not the wav): max between -1 and -4 dB. `0.0 dB` means clipped.
  Fix by capping with `loudnorm ... TP=-2.5`, never by trusting the wav peak.
- VO audibility: decode the final mp4, print RMS dB per VO window against a control window just
  after it. Aim for >= 4 dB of separation. Below that, lower the music bus (~0.40) or deepen the
  duck (~0.26) and remix; do not try to fix it with EQ.
- No silent stretch longer than ~0.8 s (an ad must never go quiet).
- A stills sheet exists and you have actually looked at it for clipped text, dead frames, missing
  product and colour mistakes.
- Frames pulled from the FINAL mp4 (not from the renderer) look right.

## Hard rules

- **Never fabricate output.** If a render, encode or measurement did not run, say so. Report the
  command you ran and the number it produced.
- **Sampling stills at cut boundaries is wrong.** That renders the transition at t_in=0 (a full
  white flash or a whip offset) and you will "fix" problems that do not exist. Sample at
  `cut + 0.45 * beat_len`, minimum 0.35 s.
- **Text at `pop=0` is scaled 1.35.** Space hero text for the scaled-up state or it will collide
  with whatever sits below it.
- **Never rotate flat 2D art in-plane to fake a spin.** Squash the width instead (SKILL.md 4.3).
  360 degree in-plane rotation reads as tumbling, not turning.
- **Pass an explicit `key=` to `blit()` for per-frame images.** The resize cache keys on `id(img)`
  and CPython reuses ids; without a key you can be served a cached resize of a different image.
- **Absolute paths in the ffmpeg concat list.** The concat demuxer resolves relative entries against
  the list file's own directory, so `chunks/list.txt` containing `chunks/chunk_0.mp4` fails with
  `Impossible to open 'chunks/chunks/chunk_0.mp4'`.
- **Do not resample 44.1 -> 48 kHz at mux time** and always pass a true-peak ceiling; lossy AAC
  overshoots the source peak and a -3.2 dBFS master decoded at 0.0 dBFS.
- **Keep the music bed well under the voice.** ~0.40 of full with a duck to 0.26 during VO. The
  default failure mode of this pipeline is a bed that drowns the voice.
- Put long scripts in `.py`/`.sh` files, not inline shell one-liners (the terminal guard blocks
  those).

## Adapting to a new brief

1. Write the VO lines per shot first and generate them (`vo.py`). Voice, then timing, then pictures.
2. Set the shot grid from a BPM: 120 BPM = 2.0 s per bar, 4-bar shots = 8.0 s. Keep every shot a
   whole number of bars so cuts land on the music.
3. Build the palette and the product geometry in the renderer. Draw the product from primitives
   (body gradient masked by a rounded shape, label band, wordmark, specular stripes, condensation).
4. Write 2-4 beats per shot with one transition each (`cut`, `flash`, `whip`, `zoom`), aiming for
   ~25-30 cuts in 60 s.
5. Run the stills pass, fix what the sheet exposes, then render, mix, and walk the gates above.
6. Deliver: `MEDIA:` path (no spaces), the shot list in the message, and every unilaterally-made
   creative decision stated explicitly. Flag any product claim that must be true before publication.

## Files

```
SKILL.md                        the playbook (read first)
scripts/validate_kit.py         execs the kit out of SKILL.md and checks beat() dispatch
example/kiki-iced-tea/
  render.py                     kit + 8 shots + 27 cuts; `stills` and `range` entry points
  vo.py                         edge-tts voiceover, silence trim, broadcast chain, timing.json
  score.py                      synth score + foley + ducking + headroom
  render_all.sh                 4 parallel chunk render, chunk boundaries on shot boundaries
  mix_and_mux.sh                concat, loudnorm true-peak pass, mux, delivery copies
  verify.py                     audio level map and frames from the final mp4
  contact_sheet.jpg             the storyboard, review artifact
```
