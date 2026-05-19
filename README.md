# Dark Room — Photo Lab

A dark, editorial-style image processing studio built with Python (Flask + OpenCV) and vanilla HTML/CSS/JS.

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Features

### 30+ Filters across 4 categories

**Colour**
- Grayscale, Sepia, Invert
- Vintage (warm fade + vignette)
- Cross Process (film-style curve shifts)
- Cyberpunk (teal/magenta colour grading)
- Duotone (custom two-colour tint)
- Warmth / Cool

**Tone**
- Brightness, Contrast, Exposure (in stops), Gamma
- Saturation, Hue Shift
- Shadows lift / Highlights crush

**Effect**
- Gaussian Blur, Motion Blur (with angle)
- Sharpen, Unsharp Mask
- Emboss, Edge Detect, Pencil Sketch
- Oil Paint (bilateral filter)
- Pixelate, Halftone dots
- Vignette, Film Noise, Posterize

**Transform**
- Rotate (arbitrary angle), Flip H/V
- Resize, Crop

### UI Highlights
- **Operations Stack** — build a pipeline of filters, reorder, remove individually
- **Compare slider** — drag to compare original vs processed
- **Undo history** — step back through processed states
- **Export** — JPEG / PNG / WEBP with quality control
- **Keyboard shortcuts** — Ctrl+Z undo, Ctrl+S export, Enter to process
- Non-destructive: original is always preserved

---

## Architecture

```
main.py          Flask API server
  POST /process  Apply filter pipeline to base64 image
  GET  /filters  List all available filters
index.html       Single-file frontend (HTML + CSS + JS)
```

All image data is transferred as base64 — no files are written to disk.
