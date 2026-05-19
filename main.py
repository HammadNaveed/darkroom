import os
import io
import base64
import json
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder=".", static_url_path="")

# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def b64_to_cv2(b64_str):
    """Decode a base-64 image string → OpenCV BGR array."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def cv2_to_b64(img, ext="JPEG", quality=92):
    """Encode OpenCV BGR array → base-64 data-URL."""
    if ext.upper() == "PNG":
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 6]
        _, buf = cv2.imencode(".png", img, encode_param)
        mime = "image/png"
    else:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buf = cv2.imencode(".jpg", img, encode_param)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def clamp(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
#  Filter implementations
# ──────────────────────────────────────────────

def apply_filter(img, name, params):
    """Dispatch to the correct filter function."""
    name = name.lower()

    # ── Colour filters ──────────────────────────
    if name == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif name == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                            [0.349, 0.686, 0.168],
                            [0.393, 0.769, 0.189]])
        out = cv2.transform(img.astype(np.float32), kernel)
        return clamp(out)

    elif name == "invert":
        return cv2.bitwise_not(img)

    elif name == "duotone":
        # Two-colour tint: shadows → colour_a, highlights → colour_b
        ca = params.get("color_a", [20, 10, 60])   # dark tone (BGR)
        cb = params.get("color_b", [255, 200, 130]) # light tone (BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = ca[c] * (1 - gray) + cb[c] * gray
        return clamp(out)

    elif name == "vintage":
        # Fade + warm lift + vignette
        out = img.astype(np.float32)
        # Warm lift (add to red/green, reduce blue)
        out[:, :, 2] = out[:, :, 2] * 1.15 + 20  # R
        out[:, :, 1] = out[:, :, 1] * 1.05 + 10  # G
        out[:, :, 0] = out[:, :, 0] * 0.85        # B
        # Fade (compress range toward midtones)
        out = out * 0.75 + 40
        # Vignette
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        mask = 1 - np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2) * 0.5
        mask = np.clip(mask, 0, 1)
        out *= mask[:, :, np.newaxis]
        return clamp(out)

    elif name == "cyberpunk":
        # Teal shadows / magenta highlights
        out = img.astype(np.float32)
        out[:, :, 0] = out[:, :, 0] * 1.3  # boost B (teal)
        out[:, :, 2] = out[:, :, 2] * 0.8  # reduce R
        # Add magenta where bright
        luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        out[:, :, 2] += luma * 80  # R up in highlights
        out[:, :, 0] -= luma * 40  # B down in highlights
        return clamp(out)

    elif name == "warmth":
        strength = params.get("strength", 50)
        out = img.astype(np.float32)
        out[:, :, 2] = out[:, :, 2] + strength        # R up
        out[:, :, 0] = out[:, :, 0] - strength * 0.3  # B down
        return clamp(out)

    elif name == "cool":
        strength = params.get("strength", 50)
        out = img.astype(np.float32)
        out[:, :, 0] = out[:, :, 0] + strength        # B up
        out[:, :, 2] = out[:, :, 2] - strength * 0.3  # R down
        return clamp(out)

    # ── Tone / exposure ─────────────────────────
    elif name == "brightness":
        val = params.get("value", 0)
        return clamp(img.astype(np.float32) + val)

    elif name == "contrast":
        val = params.get("value", 1.0)
        mean = img.mean()
        return clamp((img.astype(np.float32) - mean) * val + mean)

    elif name == "exposure":
        stops = params.get("stops", 1.0)
        factor = 2 ** stops
        return clamp(img.astype(np.float32) * factor)

    elif name == "gamma":
        g = params.get("value", 1.0)
        table = np.array([(i / 255.0) ** (1.0 / g) * 255
                           for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    elif name == "shadows":
        # Lift or crush shadows
        lift = params.get("lift", 30)
        luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        shadow_mask = np.clip(1 - luma * 2, 0, 1)[:, :, np.newaxis]
        out = img.astype(np.float32) + lift * shadow_mask
        return clamp(out)

    elif name == "highlights":
        crush = params.get("crush", -30)
        luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hi_mask = np.clip(luma * 2 - 1, 0, 1)[:, :, np.newaxis]
        out = img.astype(np.float32) + crush * hi_mask
        return clamp(out)

    elif name == "saturation":
        val = params.get("value", 1.5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * val, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    elif name == "hue_shift":
        shift = int(params.get("shift", 30))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── Blur / sharpen ──────────────────────────
    elif name == "blur":
        radius = max(1, int(params.get("radius", 5)))
        k = radius * 2 + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    elif name == "motion_blur":
        size = max(3, int(params.get("size", 15)))
        angle = params.get("angle", 0)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (size, size))
        return cv2.filter2D(img, -1, kernel)

    elif name == "sharpen":
        strength = params.get("strength", 1.0)
        kernel = np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]], dtype=np.float32)
        kernel = np.eye(3) * (1 - strength) + kernel * strength
        return clamp(cv2.filter2D(img.astype(np.float32), -1, kernel))

    elif name == "unsharp_mask":
        radius = max(1, int(params.get("radius", 3)))
        amount = params.get("amount", 1.5)
        k = radius * 2 + 1
        blurred = cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)
        out = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
        return clamp(out)

    # ── Stylistic ───────────────────────────────
    elif name == "emboss":
        kernel = np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        embossed = cv2.filter2D(gray.astype(np.float32), -1, kernel) + 128
        out = clamp(embossed)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    elif name == "edge_detect":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        inv = cv2.bitwise_not(edges)
        return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    elif name == "sketch":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, cv2.bitwise_not(blurred), scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif name == "oil_paint":
        # Bilateral filter approximation of oil painting
        for _ in range(3):
            img = cv2.bilateralFilter(img, 9, 75, 75)
        return img

    elif name == "pixelate":
        size = max(2, int(params.get("size", 12)))
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // size, h // size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    elif name == "halftone":
        # Greyscale halftone-dot look
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        dot = int(params.get("dot_size", 8))
        h, w = gray.shape
        out = np.ones((h, w), dtype=np.float32)
        for y in range(0, h, dot):
            for x in range(0, w, dot):
                patch = gray[y:y+dot, x:x+dot]
                brightness = patch.mean()
                r = int(brightness * dot / 2)
                cy, cx = y + dot // 2, x + dot // 2
                if r > 0:
                    cv2.circle(out, (cx, cy), r, 0, -1)
        out_u8 = (out * 255).astype(np.uint8)
        return cv2.cvtColor(out_u8, cv2.COLOR_GRAY2BGR)

    elif name == "vignette":
        strength = params.get("strength", 0.7)
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        mask = 1 - np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2) * strength
        mask = np.clip(mask, 0, 1)
        out = img.astype(np.float32) * mask[:, :, np.newaxis]
        return clamp(out)

    elif name == "noise":
        amount = params.get("amount", 25)
        noise = np.random.randint(-amount, amount, img.shape, dtype=np.int16)
        return clamp(img.astype(np.int16) + noise)

    elif name == "posterize":
        levels = max(2, int(params.get("levels", 4)))
        step = 255 // (levels - 1)
        table = np.array([round(i / step) * step for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    elif name == "cross_process":
        # Emulate cross-processing: aggressive curve shifts per channel
        def curve(channel, pts):
            x = np.array([p[0] for p in pts], dtype=np.float32)
            y = np.array([p[1] for p in pts], dtype=np.float32)
            table = np.interp(np.arange(256), x, y).astype(np.uint8)
            return cv2.LUT(channel, table)
        b, g, r = cv2.split(img)
        r = curve(r, [(0,0),(64,100),(192,220),(255,255)])
        g = curve(g, [(0,0),(64,50),(192,200),(255,255)])
        b = curve(b, [(0,60),(128,128),(255,200)])
        return cv2.merge([b, g, r])

    # ── Transform ───────────────────────────────
    elif name == "rotate":
        angle = params.get("angle", 90)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    elif name == "flip_h":
        return cv2.flip(img, 1)

    elif name == "flip_v":
        return cv2.flip(img, 0)

    elif name == "resize":
        w = int(params.get("width", img.shape[1]))
        h = int(params.get("height", img.shape[0]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    elif name == "crop":
        x = int(params.get("x", 0))
        y = int(params.get("y", 0))
        w = int(params.get("width", img.shape[1]))
        h = int(params.get("height", img.shape[0]))
        x2 = min(x + w, img.shape[1])
        y2 = min(y + h, img.shape[0])
        return img[y:y2, x:x2]

    # Unknown filter – return unchanged
    return img


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.get_json(force=True)
        b64 = data.get("image", "")
        operations = data.get("operations", [])   # list of {filter, params}
        export_fmt = data.get("format", "JPEG")
        quality = int(data.get("quality", 92))

        if not b64:
            return jsonify({"error": "No image provided"}), 400

        img = b64_to_cv2(b64)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        for op in operations:
            filter_name = op.get("filter", "")
            params = op.get("params", {})
            img = apply_filter(img, filter_name, params)

        result = cv2_to_b64(img, ext=export_fmt, quality=quality)
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/filters", methods=["GET"])
def list_filters():
    filters = {
        "color": ["grayscale", "sepia", "invert", "duotone", "vintage",
                  "cyberpunk", "warmth", "cool", "cross_process"],
        "tone":  ["brightness", "contrast", "exposure", "gamma",
                  "shadows", "highlights", "saturation", "hue_shift"],
        "effect":["blur", "motion_blur", "sharpen", "unsharp_mask",
                  "emboss", "edge_detect", "sketch", "oil_paint",
                  "pixelate", "halftone", "vignette", "noise",
                  "posterize"],
        "transform": ["rotate", "flip_h", "flip_v", "resize", "crop"],
    }
    return jsonify(filters)


if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║   Dark Room  –  Photo Lab Server     ║")
    print("║   http://127.0.0.1:5000              ║")
    print("╚══════════════════════════════════════╝")
    app.run(debug=True, port=5000)
