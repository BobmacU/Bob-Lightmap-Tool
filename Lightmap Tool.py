import os, cv2
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import threading
import concurrent.futures
import time

# --- Suppress noisy OpenCV/OpenEXR warning prints at FD-level ---
def _suppress_opencv_exr_warning():
    # Prefer disabling via env var first
    try:
        os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')
    except Exception:
        pass

    try:
        # Create a pipe to capture stderr FD (fd 2) and filter lines
        orig_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(orig_stderr_fd)
        rfd, wfd = os.pipe()
        # replace stderr FD with pipe write end
        os.dup2(wfd, orig_stderr_fd)
        os.close(wfd)

        def _forwarder(read_fd, write_fd):
            # read bytes from read_fd, decode, filter undesired lines, forward to saved stderr
            with os.fdopen(read_fd, 'rb', closefd=True) as r, os.fdopen(write_fd, 'wb', closefd=False) as out:
                buf = b''
                while True:
                    data = r.read(1024)
                    if not data:
                        break
                    buf += data
                    # process complete lines
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            s = line.decode('utf-8', errors='ignore')
                        except Exception:
                            s = ''
                        # filter OpenEXR/OpenCV EXR init warnings
                        if 'OpenEXR' in s or 'grfmt_exr' in s or 'OPENCV_IO_ENABLE_OPENEXR' in s:
                            # drop the line
                            continue
                        out.write(line + b'\n')
                        out.flush()
                # flush remaining
                if buf:
                    try:
                        s = buf.decode('utf-8', errors='ignore')
                    except Exception:
                        s = ''
                    if not ('OpenEXR' in s or 'grfmt_exr' in s or 'OPENCV_IO_ENABLE_OPENEXR' in s):
                        out.write(buf)
                        out.flush()

        t = threading.Thread(target=_forwarder, args=(rfd, saved_stderr_fd), daemon=True)
        t.start()
    except Exception:
        # best-effort; if we fail, ignore and continue
        pass

# Run suppression as early as possible
_suppress_opencv_exr_warning()

# optional backends
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

# Force-disable OpenCV usage to avoid OpenEXR runtime warnings; prefer imageio.
cv2 = None
CV2_AVAILABLE = False

try:
    import imageio.v3 as iio
    IMAGEIO_AVAILABLE = True
except Exception:
    iio = None
    IMAGEIO_AVAILABLE = False

# Not using native OpenEXR bindings; prefer imageio for EXR/HDR handling
OPENEXR_AVAILABLE = False
OpenEXR = None
Imath = None

# ---------------------------
# Helpers / conversion utils
# ---------------------------
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    return Path(relative_path)

# sRGB <-> linear helpers (scalar versions used by PIL fallback)
def srgb_to_linear_chan(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def linear_to_srgb_chan(c):
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055

# NumPy vectorized sRGB <-> linear
def _srgb_to_linear_np(arr):
    mask = arr <= 0.04045
    return np.where(mask, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb_np(arr):
    mask = arr <= 0.0031308
    return np.where(mask, arr * 12.92, 1.055 * (arr ** (1.0/2.4)) - 0.055)

# Note: resample selection removed — code uses sensible defaults.

def _is_hdr_ext(p: Path):
    # HDR/EXR support is disabled — treat all files as LDR.
    return False

# OpenCV support detection removed: we force use of imageio for EXR/HDR.

# Tonemappers (input: linear float32 RGB array, shape (...,3)) -> linear LDR RGB 0..1
def _tonemap_reinhard(rgb):
    # classic Reinhard; caller may apply exposure and blend with original
    return rgb / (1.0 + rgb)

def _tonemap_aces_fitted(rgb):
    # ACES approximation (Jim Hejl / Stephen Hill fit)
    a = 2.51; b = 0.03; c = 2.43; d = 0.59; e = 0.14
    x = rgb
    return np.clip((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0)


def _apply_tonemap_to_image(arr, method='Reinhard', strength=1.0, exposure=0.0):
    """
    Apply tonemap to a linear float32 RGB(A) array and return a PIL RGBA image.
    - `arr` is linear RGB(A) float32 in range [0,inf)
    - `method` is 'Reinhard', 'ACES' or 'None'
    - `strength` 0..1 blends between original linear and tonemapped result
    - `exposure` in stops (multiply linear RGB by 2**exposure before tonemapping)
    """
    if method is None or method == 'None':
        srgb = _linear_to_srgb_np(np.clip(arr[..., :3], 0.0, 1.0))
        a = arr[..., 3:4] if arr.shape[2] > 3 else np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
        out8 = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        alpha8 = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        rgba = np.concatenate([out8, alpha8], axis=2)
        return Image.fromarray(rgba, mode='RGBA')

    # copy to avoid mutating input
    lin = np.copy(arr[..., :3])
    # apply exposure (stops)
    try:
        lin = lin * (2.0 ** float(exposure))
    except Exception:
        pass

    if method == 'Reinhard':
        tm = _tonemap_reinhard(lin)
    elif method == 'ACES':
        tm = _tonemap_aces_fitted(lin)
    else:
        tm = np.clip(lin, 0.0, 1.0)

    # blend between original (clamped to [0,1]) and tonemapped result
    lin_clamped = np.clip(lin, 0.0, 1.0)
    blended = lin_clamped * (1.0 - strength) + tm * strength

    srgb = _linear_to_srgb_np(np.clip(blended, 0.0, 1.0))
    a = arr[..., 3:4] if arr.shape[2] > 3 else np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
    out8 = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    alpha8 = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    rgba = np.concatenate([out8, alpha8], axis=2)
    return Image.fromarray(rgba, mode='RGBA')

def _tonemap_filmic_uncharted2(rgb, exposure=1.0):
    # Filmic tonemapper removed.
    raise NotImplementedError("Filmic tonemapper has been removed")

def _load_any_image(path: Path):
    """Load image from disk.
    Returns tuple (is_hdr, data)
    - if is_hdr True: `data` is a numpy.float32 array shape (H,W,4) in RGB(A) linear space (no gamma)
    - if is_hdr False: `data` is a PIL.Image instance (LDR)
    """
    # HDR/EXR support removed: always return an LDR PIL Image (RGBA)
    p = Path(path)
    img = Image.open(str(p)).convert('RGBA')
    return False, img


def _resize_rgba_premult(pil_img: Image.Image, size):
    """Resize an RGBA PIL image using premultiplied-alpha to avoid halos.
    Falls back to normal resize if NumPy not available.
    Returns a PIL.Image (RGBA).
    """
    resample = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
    if not NUMPY_AVAILABLE or np is None:
        return pil_img.resize(size, resample)
    try:
        arr = (np.asarray(pil_img).astype(np.float32) / 255.0)
        if arr.ndim != 3 or arr.shape[2] not in (3,4):
            return pil_img.resize(size, resample)
        if arr.shape[2] == 3:
            a = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
            arr = np.concatenate([arr, a], axis=2)
        rgb = arr[..., :3]
        a = arr[..., 3:4]
        premult = rgb * a
        rgba = np.concatenate([premult, a], axis=2)
        tmp = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGBA')
        tmp = tmp.resize(size, resample)
        res = (np.asarray(tmp).astype(np.float32) / 255.0)
        res_a = res[..., 3:4]
        eps = 1e-6
        res_rgb = np.where(res_a > eps, res[..., :3] / np.maximum(res_a, eps), 0.0)
        out = np.concatenate([res_rgb, res_a], axis=2)
        return Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGBA')
    except Exception:
        return pil_img.resize(size, resample)

def _save_any_image(path: Path, data):
    p = Path(path)
    # HDR/EXR support removed: save as standard LDR file via PIL.
    if isinstance(data, Image.Image):
        data.save(str(p))
        return
    # numpy array -> PIL
    arr = data
    # assume linear float32 0..1 or uint8 0..255
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGBA')
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode='RGBA')
    img.save(str(p))

# ---------------------------
# Crop logic
# ---------------------------
@dataclass
class CropSettings:
    tile_scale_x: float = 1.0
    tile_scale_y: float = 1.0
    offset_x: float = 1.0
    offset_y: float = 1.0
    output_name: str = "cropped_lightmap.png"
    output_dir: str = None
    out_width: int = None
    out_height: int = None

class LightmapCropper:
    def crop_image(self, image: Image.Image, s: CropSettings):
        width, height = image.size
        # TILE first, then OFFSET (user requested)
        left = int(s.offset_x * width)
        top = int((1 - s.offset_y - s.tile_scale_y) * height)
        right = int((s.offset_x + s.tile_scale_x) * width)
        bottom = int((1 - s.offset_y) * height)
        left = max(0, left); top = max(0, top); right = min(width, right); bottom = min(height, bottom)
        return image.crop((left, top, right, bottom))

    def crop_file(self, path, s: CropSettings):
        is_hdr, data = _load_any_image(Path(path))
        left = int(s.offset_x * (data.shape[1] if is_hdr else Image.open(path).size[0]))
        top = int((1 - s.offset_y - s.tile_scale_y) * (data.shape[0] if is_hdr else Image.open(path).size[1]))
        right = int((s.offset_x + s.tile_scale_x) * (data.shape[1] if is_hdr else Image.open(path).size[0]))
        bottom = int((1 - s.offset_y) * (data.shape[0] if is_hdr else Image.open(path).size[1]))
        if is_hdr:
            h, w = data.shape[0], data.shape[1]
            left = max(0, left); top = max(0, top); right = min(w, right); bottom = min(h, bottom)
            cropped = data[top:bottom, left:right, :]
            if s.out_width and s.out_height:
                # resize via PIL (clamps HDR values) since OpenCV is disabled
                pil_tmp = Image.fromarray((np.clip(cropped[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB')
                pil_a = Image.fromarray((np.clip(cropped[..., 3]*255.0, 0, 255).astype(np.uint8)), mode='L')
                pil_tmp.putalpha(pil_a)
                pil_tmp = pil_tmp.resize((s.out_width, s.out_height), getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
                cropped = (np.asarray(pil_tmp).astype(np.float32) / 255.0)
            out_dir = s.output_dir or (Path(path).parent / "Cropped")
            os.makedirs(out_dir, exist_ok=True)
            out_path = Path(out_dir) / s.output_name
            _save_any_image(out_path, cropped)
            return out_path
        else:
            img = data if isinstance(data, Image.Image) else Image.open(path)
            cropped = self.crop_image(img, s)
            if s.out_width and s.out_height:
                resample = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                cropped = cropped.resize((s.out_width, s.out_height), resample)
            out_dir = s.output_dir or (Path(path).parent / "Cropped")
            os.makedirs(out_dir, exist_ok=True)
            out_path = Path(out_dir) / s.output_name
            cropped.save(out_path)
            return out_path

# ---------------------------
# Compose (core) — clean pipeline
# Priority: torch -> numpy -> PIL fallback
# ---------------------------
CHUNK_THRESHOLD_PIXELS = 20_000_000  # chunk if bigger than this for numpy path

def _compose_linear_numpy(base_rgb, layer_rgb, mode, intensity):
    if mode == "multiply":
        comp = base_rgb * layer_rgb
    elif mode == "add":
        comp = np.clip(base_rgb + layer_rgb, 0.0, 1.0)
    elif mode in ("mix","blend"):
        comp = layer_rgb
    else:
        raise ValueError("Unsupported mode")
    return base_rgb * (1.0 - intensity) + comp * intensity

def _compose_linear_torch(b, l, mode, intensity):
    if mode == "multiply":
        comp = b * l
    elif mode == "add":
        comp = torch.clamp(b + l, 0.0, 1.0)
    elif mode in ("mix","blend"):
        comp = l
    else:
        raise ValueError("Unsupported mode")
    return b * (1.0 - intensity) + comp * intensity

def compose_images(base_img: Image.Image, layer_img: Image.Image, mode: str, intensity: float,
                   use_linear: bool, use_torch: bool) -> Image.Image:
    """
    Compose two RGBA images. Diffuse is the top layer (layer_img). Lightmap is base_img.
    This function ensures base and layer sizes match (resizing base to match layer if needed).
    """
    intensity = float(max(0.0, min(1.0, intensity)))
    m = (mode or "multiply").lower()

    base = base_img.convert("RGBA")
    layer = layer_img.convert("RGBA")
    if base.size != layer.size:
        # resize base to match layer using premultiplied-alpha to avoid halos
        base = _resize_rgba_premult(base, layer.size)

    W, H = base.size
    total_pixels = W * H

    # 1) Torch path
    if use_torch and TORCH_AVAILABLE:
        try:
            # detect device at runtime so installing/enabling CUDA later works
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # prepare numpy arrays (linear or sRGB handling is performed below)
            b_arr = (np.asarray(base).astype(np.float32) / 255.0)
            l_arr = (np.asarray(layer).astype(np.float32) / 255.0)
            H, W = b_arr.shape[0], b_arr.shape[1]

            # For large images, process in chunks (rows) to limit peak GPU memory.
            GPU_CHUNK_PIXELS = max(4_000_000, CHUNK_THRESHOLD_PIXELS // 5)
            if device.type == 'cuda' and (W * H) > GPU_CHUNK_PIXELS:
                approx_h = max(1, int(GPU_CHUNK_PIXELS / W))
                out_arr = np.zeros((H, W, 4), dtype=np.uint8)
                for y0 in range(0, H, approx_h):
                    y1 = min(H, y0 + approx_h)
                    b_chunk = b_arr[y0:y1, :, :]
                    l_chunk = l_arr[y0:y1, :, :]
                    b_rgb = torch.from_numpy(b_chunk[..., :3]).to(device)
                    l_rgb = torch.from_numpy(l_chunk[..., :3]).to(device)
                    b_a = torch.from_numpy(b_chunk[..., 3:4]).to(device)
                    l_a = torch.from_numpy(l_chunk[..., 3:4]).to(device)

                    if use_linear:
                        b_mask = b_rgb <= 0.04045
                        l_mask = l_rgb <= 0.04045
                        b_lin = torch.where(b_mask, b_rgb / 12.92, ((b_rgb + 0.055) / 1.055) ** 2.4)
                        l_lin = torch.where(l_mask, l_rgb / 12.92, ((l_rgb + 0.055) / 1.055) ** 2.4)
                        comp_lin = _compose_linear_torch(b_lin, l_lin, m, 1.0)
                        alpha_eff = l_a * intensity
                        out_lin = b_lin * (1.0 - alpha_eff) + comp_lin * alpha_eff
                        out_a = b_a + l_a * intensity * (1.0 - b_a)
                        mask_res = out_lin <= 0.0031308
                        res_srgb = torch.where(mask_res, out_lin * 12.92, 1.055 * (out_lin ** (1.0 / 2.4)) - 0.055)
                        out = torch.cat([res_srgb, out_a], dim=2)
                    else:
                        if m == "multiply":
                            comp = b_rgb * l_rgb
                        elif m == "add":
                            comp = torch.clamp(b_rgb + l_rgb, 0.0, 1.0)
                        else:
                            comp = l_rgb
                        alpha_eff = l_a * intensity
                        out_rgb = b_rgb * (1.0 - alpha_eff) + comp * alpha_eff
                        out_a = b_a + l_a * intensity * (1.0 - b_a)
                        out = torch.cat([out_rgb, out_a], dim=2)

                    out = torch.clamp(out, 0.0, 1.0)
                    out_np = (out.cpu().numpy() * 255.0).round().astype(np.uint8)
                    out_arr[y0:y1, :, :] = out_np
                return Image.fromarray(out_arr, mode="RGBA")
            else:
                # single-shot (fits comfortably) path
                b_rgb = torch.from_numpy(b_arr[..., :3]).to(device)
                l_rgb = torch.from_numpy(l_arr[..., :3]).to(device)
                b_a = torch.from_numpy(b_arr[..., 3:4]).to(device)
                l_a = torch.from_numpy(l_arr[..., 3:4]).to(device)

                if use_linear:
                    b_mask = b_rgb <= 0.04045
                    l_mask = l_rgb <= 0.04045
                    b_lin = torch.where(b_mask, b_rgb / 12.92, ((b_rgb + 0.055) / 1.055) ** 2.4)
                    l_lin = torch.where(l_mask, l_rgb / 12.92, ((l_rgb + 0.055) / 1.055) ** 2.4)
                    comp_lin = _compose_linear_torch(b_lin, l_lin, m, 1.0)
                    alpha_eff = l_a * intensity
                    out_lin = b_lin * (1.0 - alpha_eff) + comp_lin * alpha_eff
                    out_a = b_a + l_a * intensity * (1.0 - b_a)
                    mask_res = out_lin <= 0.0031308
                    res_srgb = torch.where(mask_res, out_lin * 12.92, 1.055 * (out_lin ** (1.0 / 2.4)) - 0.055)
                    out = torch.cat([res_srgb, out_a], dim=2)
                else:
                    if m == "multiply":
                        comp = b_rgb * l_rgb
                    elif m == "add":
                        comp = torch.clamp(b_rgb + l_rgb, 0.0, 1.0)
                    else:
                        comp = l_rgb
                    alpha_eff = l_a * intensity
                    out_rgb = b_rgb * (1.0 - alpha_eff) + comp * alpha_eff
                    out_a = b_a + l_a * intensity * (1.0 - b_a)
                    out = torch.cat([out_rgb, out_a], dim=2)

                out = torch.clamp(out, 0.0, 1.0)
                out_np = (out.cpu().numpy() * 255.0).round().astype(np.uint8)
                return Image.fromarray(out_np, mode="RGBA")
        except Exception as e:
            # fallback to next path
            print("[compose_images] torch path failed:", e)

    # 2) NumPy fast path (supports chunking)
    if NUMPY_AVAILABLE and np is not None:
        try:
            if total_pixels > CHUNK_THRESHOLD_PIXELS:
                # chunk by rows
                approx_chunk_h = max(1, int(CHUNK_THRESHOLD_PIXELS / W))
                out_arr = np.zeros((H, W, 4), dtype=np.uint8)
                for y0 in range(0, H, approx_chunk_h):
                    y1 = min(H, y0 + approx_chunk_h)
                    b_chunk = (np.asarray(base.crop((0, y0, W, y1))).astype(np.float32) / 255.0)
                    l_chunk = (np.asarray(layer.crop((0, y0, W, y1))).astype(np.float32) / 255.0)
                    b_rgb = b_chunk[..., :3]; l_rgb = l_chunk[..., :3]
                    b_a = b_chunk[..., 3:4]; l_a = l_chunk[..., 3:4]
                    if use_linear:
                        b_lin = _srgb_to_linear_np(b_rgb)
                        l_lin = _srgb_to_linear_np(l_rgb)
                        comp_lin = _compose_linear_numpy(b_lin, l_lin, m, 1.0)
                        alpha_eff = l_a * intensity
                        out_lin = b_lin * (1.0 - alpha_eff) + comp_lin * alpha_eff
                        out_a = b_a + l_a * intensity * (1.0 - b_a)
                        res_srgb = _linear_to_srgb_np(out_lin)
                        out_chunk = np.concatenate([res_srgb, out_a], axis=2)
                    else:
                        if m == "multiply":
                            comp = b_rgb * l_rgb
                        elif m == "add":
                            comp = np.clip(b_rgb + l_rgb, 0.0, 1.0)
                        else:
                            comp = l_rgb
                        alpha_eff = l_a * intensity
                        out_rgb = b_rgb * (1.0 - alpha_eff) + comp * alpha_eff
                        out_a = b_a + l_a * intensity * (1.0 - b_a)
                        out_chunk = np.concatenate([out_rgb, out_a], axis=2)
                    out_chunk = np.clip(out_chunk, 0.0, 1.0)
                    out_arr[y0:y1, :, :] = (out_chunk * 255.0).round().astype(np.uint8)
                return Image.fromarray(out_arr, mode="RGBA")
            else:
                b_arr = (np.asarray(base).astype(np.float32) / 255.0)
                l_arr = (np.asarray(layer).astype(np.float32) / 255.0)
                b_rgb = b_arr[..., :3]; l_rgb = l_arr[..., :3]
                b_a = b_arr[..., 3:4]; l_a = l_arr[..., 3:4]
                if use_linear:
                    b_lin = _srgb_to_linear_np(b_rgb)
                    l_lin = _srgb_to_linear_np(l_rgb)
                    comp_lin = _compose_linear_numpy(b_lin, l_lin, m, 1.0)
                    alpha_eff = l_a * intensity
                    out_lin = b_lin * (1.0 - alpha_eff) + comp_lin * alpha_eff
                    out_a = b_a + l_a * intensity * (1.0 - b_a)
                    res_srgb = _linear_to_srgb_np(out_lin)
                    out = np.concatenate([res_srgb, out_a], axis=2)
                else:
                    if m == "multiply":
                        comp = b_rgb * l_rgb
                    elif m == "add":
                        comp = np.clip(b_rgb + l_rgb, 0.0, 1.0)
                    else:
                        comp = l_rgb
                    alpha_eff = l_a * intensity
                    out_rgb = b_rgb * (1.0 - alpha_eff) + comp * alpha_eff
                    out_a = b_a + l_a * intensity * (1.0 - b_a)
                    out = np.concatenate([out_rgb, out_a], axis=2)
                out = np.clip(out, 0.0, 1.0)
                out8 = (out * 255.0).round().astype(np.uint8)
                return Image.fromarray(out8, mode="RGBA")
        except Exception as e:
            print("[compose_images] numpy path failed:", e)

    # 3) PIL per-pixel fallback (slow but correct)
    # convert to floats and do per-pixel arithmetic
    bpx = list(base.getdata())
    lpx = list(layer.getdata())
    out_pixels = []
    for (br, bg, bb, ba), (lr, lg, lb, la) in zip(bpx, lpx):
        brf = br / 255.0; bgf = bg / 255.0; bbf = bb / 255.0; baf = ba / 255.0
        lrf = lr / 255.0; lgf = lg / 255.0; lbf = lb / 255.0; laf = la / 255.0
        if use_linear:
            br_lin = srgb_to_linear_chan(brf); bg_lin = srgb_to_linear_chan(bgf); bb_lin = srgb_to_linear_chan(bbf)
            lr_lin = srgb_to_linear_chan(lrf); lg_lin = srgb_to_linear_chan(lgf); lb_lin = srgb_to_linear_chan(lbf)
            if m == "multiply":
                cr_lin = br_lin * lr_lin; cg_lin = bg_lin * lg_lin; cb_lin = bb_lin * lb_lin
            elif m == "add":
                cr_lin = min(1.0, br_lin + lr_lin); cg_lin = min(1.0, bg_lin + lg_lin); cb_lin = min(1.0, bb_lin + lb_lin)
            else:
                cr_lin, cg_lin, cb_lin = lr_lin, lg_lin, lb_lin
            alpha_eff = laf * intensity
            out_r_lin = br_lin * (1.0 - alpha_eff) + cr_lin * alpha_eff
            out_g_lin = bg_lin * (1.0 - alpha_eff) + cg_lin * alpha_eff
            out_b_lin = bb_lin * (1.0 - alpha_eff) + cb_lin * alpha_eff
            out_a = baf + laf * intensity * (1.0 - baf)
            out_r = linear_to_srgb_chan(out_r_lin); out_g = linear_to_srgb_chan(out_g_lin); out_b = linear_to_srgb_chan(out_b_lin)
            out_pixels.append((int(round(out_r*255)), int(round(out_g*255)), int(round(out_b*255)), int(round(out_a*255))))
        else:
            if m == "multiply":
                cr = brf * lrf; cg = bgf * lgf; cb = bbf * lbf
            elif m == "add":
                cr = min(1.0, brf + lrf); cg = min(1.0, bgf + lgf); cb = min(1.0, bbf + lbf)
            else:
                cr, cg, cb = lrf, lgf, lbf
            alpha_eff = laf * intensity
            out_r = brf * (1.0 - alpha_eff) + cr * alpha_eff
            out_g = bgf * (1.0 - alpha_eff) + cg * alpha_eff
            out_b = bbf * (1.0 - alpha_eff) + cb * alpha_eff
            out_a = baf + laf * intensity * (1.0 - baf)
            out_pixels.append((int(round(out_r*255)), int(round(out_g*255)), int(round(out_b*255)), int(round(out_a*255))))
    out_img = Image.new("RGBA", base.size)
    out_img.putdata(out_pixels)
    return out_img

def compose_files_to_image(diffuse_path: Path, lightmap_path: Path, mode: str = "multiply",
                           intensity: float = 1.0, resize_lm_to_diffuse: bool = False,
                           use_linear: bool = False, treat_lm_as_linear: bool = False,
                           diffuse_over_lm: bool = False, use_torch: bool = False,
                           srgb_multiply: bool = False,
                           ):
    """
    Compose files: diffuse (top) and lightmap (base).
    - resize_lm_to_diffuse: whether to scale the lightmap to match the diffuse
    - treat_lm_as_linear: interpret incoming lightmap channels as linear when composing in sRGB path
    """
    # New linear-first pipeline (LDR only). If NumPy isn't available, fall back to old PIL compose.
    if not NUMPY_AVAILABLE or np is None:
        # fallback: previous behavior using PIL compose — but use premultiplied-alpha resize
        diffuse = Image.open(diffuse_path).convert("RGBA")
        lm = Image.open(lightmap_path).convert("RGBA")
        if resize_lm_to_diffuse:
            lm = _resize_rgba_premult(lm, diffuse.size)
        if diffuse_over_lm:
            return compose_images(diffuse, lm, mode.lower(), intensity, use_linear, use_torch)
        return compose_images(lm, diffuse, mode.lower(), intensity, use_linear, use_torch)

    # Load as PIL then convert to float32 linear arrays
    diff_img = Image.open(diffuse_path).convert('RGBA')
    lm_img = Image.open(lightmap_path).convert('RGBA')

    d_arr = (np.asarray(diff_img).astype(np.float32) / 255.0)
    l_arr = (np.asarray(lm_img).astype(np.float32) / 255.0)

    # ensure alpha channel exists
    if d_arr.shape[2] == 3:
        d_arr = np.concatenate([d_arr, np.ones((d_arr.shape[0], d_arr.shape[1], 1), dtype=np.float32)], axis=2)
    if l_arr.shape[2] == 3:
        l_arr = np.concatenate([l_arr, np.ones((l_arr.shape[0], l_arr.shape[1], 1), dtype=np.float32)], axis=2)

    # Convert sRGB -> linear for diffuse and lightmap
    # (we always convert the loaded 8-bit lightmap to linear space; resizing is done
    #  via a premultiplied sRGB round-trip to avoid halos while preserving color)
    d_arr[..., :3] = _srgb_to_linear_np(d_arr[..., :3])
    l_arr[..., :3] = _srgb_to_linear_np(l_arr[..., :3])

    # Bleed / fill transparent texels on the lightmap using a nearest-source fill
    # (multi-source BFS). This preserves the exact color of the nearest opaque texel
    # instead of averaging neighbors, which can introduce halos when resampling.
    def _bleed_fill(arr):
        rgb = arr[..., :3]
        a = arr[..., 3]
        H, W = a.shape
        opaque = (a > 1e-4)
        if opaque.all():
            return arr
        from collections import deque
        visited = opaque.copy()
        out_rgb = np.zeros_like(rgb)
        out_rgb[opaque] = rgb[opaque]
        q = deque()
        # seed queue with all opaque pixel coords
        ys, xs = np.nonzero(opaque)
        for y, x in zip(ys, xs):
            q.append((y, x))

        # 8-neighborhood offsets
        neigh = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        while q:
            y, x = q.popleft()
            col = out_rgb[y, x]
            for dy, dx in neigh:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    out_rgb[ny, nx] = col
                    visited[ny, nx] = True
                    q.append((ny, nx))

        arr[..., :3] = out_rgb
        # Preserve the original alpha for pixels that were transparent.
        # We fill RGB with the nearest opaque color to avoid resampling halos,
        # but we must keep the original alpha (usually 0) so the lightmap
        # doesn't become fully opaque everywhere and leak when composing.
        arr[..., 3] = a
        return arr

    l_arr = _bleed_fill(l_arr)

    # Force alpha policy: clamp alpha to [0,1]
    d_arr[..., 3] = np.clip(d_arr[..., 3], 0.0, 1.0)
    l_arr[..., 3] = np.clip(l_arr[..., 3], 0.0, 1.0)

    # Resize lightmap to diffuse (do this in linear space via PIL round-trip)
    Hd, Wd = d_arr.shape[0], d_arr.shape[1]
    Hl, Wl = l_arr.shape[0], l_arr.shape[1]
    if resize_lm_to_diffuse and (Wd != Wl or Hd != Hl):
        # Resize the lightmap using a premultiplied sRGB round-trip to avoid halos
        # while preserving linear color correctness.
        resample = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        try:
            # Convert linear->sRGB for accurate byte-domain resampling
            sRGB = _linear_to_srgb_np(np.clip(l_arr[..., :3], 0.0, 1.0))
            a = np.clip(l_arr[..., 3:4], 0.0, 1.0)
            # For resizing, use a temporary alpha that treats transparent
            # pixels as opaque (alpha=1.0) so the premultiplied resample
            # retains nearby colors and avoids halos. We do NOT write this
            # fake alpha back to `l_arr` — `l_arr` keeps the original alpha
            # (see _bleed_fill). This prevents the lightmap from leaking
            # influence across the whole texture during composition.
            a_for_premult = np.where(a > 1e-6, a, 1.0)
            premult = sRGB * a_for_premult
            rgba = np.concatenate([premult, a_for_premult], axis=2)
            tmp = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGBA')
            tmp = tmp.resize((Wd, Hd), resample)
            res = (np.asarray(tmp).astype(np.float32) / 255.0)
            res_a = res[..., 3:4]
            eps = 1e-6
            res_rgb_srgb = np.where(res_a > eps, res[..., :3] / np.maximum(res_a, eps), 0.0)
            # convert back sRGB->linear
            res_rgb_lin = _srgb_to_linear_np(np.clip(res_rgb_srgb, 0.0, 1.0))
            l_arr = np.concatenate([res_rgb_lin, res_a], axis=2)
        except Exception:
            # fallback to simple resize (less ideal)
            tmp_rgb = Image.fromarray((np.clip(l_arr[..., :3], 0.0, 1.0) * 255.0).round().astype(np.uint8), mode='RGB')
            tmp_a = Image.fromarray((np.clip(l_arr[..., 3]*255.0, 0, 255).round().astype(np.uint8)), mode='L')
            tmp_rgb.putalpha(tmp_a)
            tmp_rgb = tmp_rgb.resize((Wd, Hd), resample)
            l_arr = (np.asarray(tmp_rgb).astype(np.float32) / 255.0)

    # Normalize intensity (0.0..1.0). Do NOT directly darken the lightmap RGB here;
    # the compose step should use the lightmap alpha scaled by intensity so
    # the blending behavior matches `compose_images` and the preview.
    intensity = float(max(0.0, min(1.0, intensity)))

    # Compose in linear: treat lightmap as base, diffuse as layer to be modulated
    m = (mode or 'multiply').lower()
    base_rgb = l_arr[..., :3]
    layer_rgb = d_arr[..., :3]
    base_a = l_arr[..., 3]
    layer_a = d_arr[..., 3]

    # Attempt a PyTorch-accelerated path when requested and available.
    # This computes the same linear-space composition as the NumPy path
    # but executes on the selected device (GPU if available).
    if use_torch and TORCH_AVAILABLE and torch is not None:
        try:
            device = TORCH_DEVICE or torch.device("cpu")
            # move linear float32 arrays to torch on device
            b_t = torch.from_numpy(base_rgb.astype('float32')).to(device=device)
            l_t = torch.from_numpy(layer_rgb.astype('float32')).to(device=device)
            ba_t = torch.from_numpy(base_a.astype('float32')).to(device=device)
            la_t = torch.from_numpy(layer_a.astype('float32')).to(device=device)

            if m == 'multiply' and srgb_multiply:
                # perform multiply in sRGB domain (Unity-like): convert linear->sRGB, multiply, convert back
                def lin_to_srgb_t(x):
                    mask = x <= 0.0031308
                    return torch.where(mask, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)
                def srgb_to_lin_t(x):
                    mask = x <= 0.04045
                    return torch.where(mask, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
                b_s = lin_to_srgb_t(torch.clamp(b_t, 0.0, 1.0))
                l_s = lin_to_srgb_t(torch.clamp(l_t, 0.0, 1.0))
                comp_t = srgb_to_lin_t(torch.clamp(b_s * l_s, 0.0, 1.0))
            elif m == 'multiply':
                comp_t = b_t * l_t
            elif m == 'add':
                comp_t = torch.clamp(b_t + l_t, 0.0, 1.0)
            else:
                comp_t = l_t

            alpha_eff_t = ba_t.unsqueeze(2) * intensity
            out_rgb_t = l_t * (1.0 - alpha_eff_t) + comp_t * alpha_eff_t
            out_a_t = la_t + ba_t * (1.0 - la_t)
            out_t = torch.cat([out_rgb_t, out_a_t.unsqueeze(2)], dim=2)
            out_t = torch.clamp(out_t, 0.0, None)

            out_np = out_t.cpu().numpy()
            # Convert linear->sRGB for return as PIL Image (LDR)
            srgb = _linear_to_srgb_np(np.clip(out_np[..., :3], 0.0, 1.0))
            alpha = np.clip(out_np[..., 3:4], 0.0, 1.0)
            out8 = (np.concatenate([srgb, alpha], axis=2) * 255.0).round().astype(np.uint8)
            return Image.fromarray(out8, mode='RGBA')
        except Exception as e:
            print("[compose_files_to_image] torch accelerated path failed:", e)
            # fall back to NumPy path below

    if m == 'multiply' and srgb_multiply:
        # perform multiply in sRGB domain: linear->sRGB, multiply, then convert back to linear
        b_srgb = _linear_to_srgb_np(np.clip(base_rgb, 0.0, 1.0))
        l_srgb = _linear_to_srgb_np(np.clip(layer_rgb, 0.0, 1.0))
        comp = _srgb_to_linear_np(np.clip(b_srgb * l_srgb, 0.0, 1.0))
    elif m == 'multiply':
        comp = base_rgb * layer_rgb
    elif m == 'add':
        comp = base_rgb + layer_rgb
    else:
        comp = layer_rgb

    # Use the lightmap alpha scaled by intensity to control how much the
    # lightmap affects the diffuse. This mirrors the `compose_images` logic
    # (which computes `alpha_eff = l_a * intensity`) and avoids producing
    # a dark/black result when intensity < 1.0.
    alpha_eff = base_a[..., None] * intensity  # use lightmap alpha * intensity
    out_rgb = layer_rgb * (1.0 - alpha_eff) + comp * alpha_eff
    out_a = layer_a + base_a * (1.0 - layer_a)

    out = np.concatenate([out_rgb, out_a[..., None]], axis=2)

    # Clamp linear result
    out[..., :3] = np.clip(out[..., :3], 0.0, None)

    # Convert linear -> sRGB for return as PIL Image (LDR)
    srgb = _linear_to_srgb_np(np.clip(out[..., :3], 0.0, 1.0))
    alpha = np.clip(out[..., 3:4], 0.0, 1.0)
    out8 = (np.concatenate([srgb, alpha], axis=2) * 255.0).round().astype(np.uint8)
    return Image.fromarray(out8, mode='RGBA')

def apply_lightmap_to_diffuse(diffuse_path: Path, lightmap_path: Path, out_path: Path,
                              mode: str = "multiply", intensity: float = 1.0,
                              resize_lm_to_diffuse: bool = False,
                              use_linear: bool = False, treat_lm_as_linear: bool = False,
                              diffuse_over_lm: bool = False, use_torch: bool = False) -> (Path, object):
    result = compose_files_to_image(diffuse_path, lightmap_path, mode, intensity,
                                    resize_lm_to_diffuse, use_linear, treat_lm_as_linear,
                                    diffuse_over_lm, use_torch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # result may be a numpy float32 array (HDR) or a PIL.Image (LDR)
    if isinstance(result, np.ndarray):
        _save_any_image(out_path, result)
    else:
        result.save(str(out_path))
    return out_path, result

# ---------------------------
# GUI
# ---------------------------
class LightmapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lightmap Tool")
        self.root.geometry("980x820")
        self.root.minsize(850, 600)
        self.root.configure(bg="#1d1f21")

        # colors
        self.bg = "#1d1f21"; self.fg = "#c5c8c6"; self.entry_bg = "#2a2d2f"
        self.btn_bg = "#5f8dd3"; self.btn_fg = "white"

        # state
        self.crop_image_path = None; self.crop_output_dir = None
        self.diffuse_path = None; self.lm_path = None; self.apply_output_dir = None
        # previously allowed treating lightmap as pre-linear; option removed
        self.use_torch_var = IntVar(value=1 if TORCH_AVAILABLE else 0)
        self.use_linear_var = IntVar(value=1)  # default to linear pipeline (physically-correct)
        self.resize_var = IntVar(value=1)
        self.mode_var = StringVar(value="multiply")
        self.intensity_var = DoubleVar(value=100.0)  # 0..100 for slider
        self.tonemap_choice = StringVar(value="Reinhard")
        self.tonemap_strength_var = DoubleVar(value=100.0)  # 0..100 blending of tonemap
        self.tonemap_exposure_var = DoubleVar(value=0.0)    # exposure in stops
        # Unity multiply option removed — keep pipeline defaults
        self._preview_debounce_after = None

        # preview cache
        self._last_preview_pil = None
        self._last_preview_key = None

        # text console
        self.console = None

        self._build_ui()
        self._setup_shortcuts()

        # thread pool for apply/export
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def log(self, msg):
        if not self.console:
            return
        self.console.configure(state="normal"); self.console.insert(END, msg+"\n"); self.console.see(END); self.console.configure(state="disabled")

    # ---------- UI helpers ----------
    def _hover_button(self, btn, color, hover_color):
        def on_enter(e): btn['bg'] = hover_color
        def on_leave(e): btn['bg'] = color
        btn.bind("<Enter>", on_enter); btn.bind("<Leave>", on_leave)

    def _make_scrollable(self, parent):
        container = Frame(parent, bg=self.bg)
        container.pack(fill=BOTH, expand=True)
        canvas = Canvas(container, bg=self.bg, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient=VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        inner = Frame(canvas, bg=self.bg)
        window_id = canvas.create_window((0,0), window=inner, anchor='nw')

        def _on_inner_config(evt):
            canvas.configure(scrollregion=canvas.bbox('all'))
        inner.bind('<Configure>', _on_inner_config)

        def _on_canvas_config(evt):
            try:
                canvas.itemconfig(window_id, width=evt.width)
            except Exception:
                pass
        canvas.bind('<Configure>', _on_canvas_config)

        # mousewheel support
        def _on_mousewheel(ev):
            try:
                canvas.yview_scroll(int(-1*(ev.delta/120)), 'units')
            except Exception:
                pass
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        return container, inner

    def _build_ui(self):
        style = ttk.Style(); style.theme_use("clam")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=BOTH, expand=True, padx=8, pady=8)

        # Crop tab
        crop_frame = Frame(self.notebook, bg=self.bg)
        self._build_crop_tab(crop_frame)
        self.notebook.add(crop_frame, text="Lightmap Crop")

        # Apply tab
        apply_frame = Frame(self.notebook, bg=self.bg)
        self._build_apply_tab(apply_frame)
        self.notebook.add(apply_frame, text="Lightmap Apply")

        # Console
        self.console = Text(self.root, height=10, width=140, bg=self.entry_bg, fg=self.fg, insertbackground=self.fg)
        self.console.pack(fill=BOTH, padx=8, pady=(0,8))
        self.console.configure(state="disabled")

    # ---------- Crop tab ----------
    def _build_crop_tab(self, parent):
        frame = parent
        padx = 8; pady = 6

        def add_label_bold(text):
            lbl = Label(frame, text=text, bg=self.bg, fg=self.fg, font=("Segoe UI", 10, "bold"))
            lbl.pack(anchor="w", padx=padx, pady=(pady//2,2)); return lbl
        def add_label(text):
            lbl = Label(frame, text=text, bg=self.bg, fg=self.fg, font=("Segoe UI", 10)); lbl.pack(anchor="w", padx=padx, pady=(pady//2,2)); return lbl
        def add_button(text, cmd, color=None, hover=None):
            color = color or self.btn_bg; hover = hover or "#73a0f5"
            b = Button(frame, text=text, command=cmd, bg=color, fg=self.btn_fg, relief=FLAT, font=("Segoe UI", 10, "bold"))
            b.pack(fill=X, padx=padx, pady=6); self._hover_button(b, color, hover); return b

        add_label_bold("Lightmap File")
        add_button("Select Lightmap...", self._crop_pick_lightmap)
        self.crop_loaded_label = Label(frame, text="Loaded: None", bg=self.bg, fg=self.fg); self.crop_loaded_label.pack(anchor="w", padx=16, pady=(2,6))

        add_label("Tile Scale X")
        self.crop_scalx = Entry(frame, bg=self.entry_bg, fg=self.fg); self.crop_scalx.insert(0,"1.0"); self.crop_scalx.pack(fill=X, padx=padx, pady=2)
        add_label("Tile Scale Y")
        self.crop_scaly = Entry(frame, bg=self.entry_bg, fg=self.fg); self.crop_scaly.insert(0,"1.0"); self.crop_scaly.pack(fill=X, padx=padx, pady=2)

        add_label("Offset X")
        self.crop_offx = Entry(frame, bg=self.entry_bg, fg=self.fg); self.crop_offx.insert(0,"1.0"); self.crop_offx.pack(fill=X, padx=padx, pady=2)
        add_label("Offset Y")
        self.crop_offy = Entry(frame, bg=self.entry_bg, fg=self.fg); self.crop_offy.insert(0,"1.0"); self.crop_offy.pack(fill=X, padx=padx, pady=2)

        add_label_bold("Output Folder")
        add_button("Select Output Folder", self._crop_pick_output_folder)

        add_label("Output Filename")
        self.crop_out_name = Entry(frame, bg=self.entry_bg, fg=self.fg); self.crop_out_name.insert(0, "cropped_lightmap.png"); self.crop_out_name.pack(fill=X, padx=padx, pady=2)

        add_label("Output Width (px) — leave blank or 0 to keep original")
        self.crop_out_w_var = StringVar(value="0"); w_entry = Entry(frame, textvariable=self.crop_out_w_var, bg=self.entry_bg, fg=self.fg); w_entry.pack(fill=X, padx=padx, pady=2)
        add_label("Output Height (px) — leave blank or 0 to keep original")
        self.crop_out_h_var = StringVar(value="0"); h_entry = Entry(frame, textvariable=self.crop_out_h_var, bg=self.entry_bg, fg=self.fg); h_entry.pack(fill=X, padx=padx, pady=2)

        add_button("CROP LIGHTMAP", self._do_crop, color="#4CAF50", hover="#66bb6a")

    def _crop_pick_lightmap(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.tga;*.bmp")])
        if p:
            self.crop_image_path = p
            self.crop_loaded_label.config(text=f"Loaded: {Path(p).name}")
            self.log(f"[Crop] Selected lightmap: {p}")
            # Auto-detect autofill for crop region removed — user will set sizes manually.

    def _crop_pick_output_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.crop_output_dir = p; self.log(f"[Crop] Output folder: {p}")

    def _do_crop(self):
        if not self.crop_image_path:
            messagebox.showerror("Error", "Pick a lightmap first!")
            return
        try:
            out_w = None; out_h = None
            w_text = self.crop_out_w_var.get().strip(); h_text = self.crop_out_h_var.get().strip()
            if w_text not in ("", "0"): out_w = int(w_text)
            if h_text not in ("", "0"): out_h = int(h_text)
            out_name = self.crop_out_name.get().strip() or "cropped_lightmap.png"
            settings = CropSettings(
                tile_scale_x=float(self.crop_scalx.get()),
                tile_scale_y=float(self.crop_scaly.get()),
                offset_x=float(self.crop_offx.get()),
                offset_y=float(self.crop_offy.get()),
                output_name=out_name,
                output_dir=self.crop_output_dir,
                out_width=out_w, out_height=out_h
            )
            out = LightmapCropper().crop_file(self.crop_image_path, settings)
            self.log(f"[Crop] Saved: {out}")
            messagebox.showinfo("Success", f"Lightmap cropped and saved:\n{out}")
        except Exception as e:
            messagebox.showerror("Error", str(e)); self.log(f"[Crop] Error: {e}")

    # ---------- Apply tab ----------
    def _build_apply_tab(self, parent):
        container, frame = self._make_scrollable(parent)
        padx = 8; pady = 6

        def add_label_bold(p, text):
            lbl = Label(p, text=text, bg=self.bg, fg=self.fg, font=("Segoe UI", 10, "bold"))
            lbl.pack(anchor="w", padx=padx, pady=(pady//2,2)); return lbl
        def add_label(p, text):
            lbl = Label(p, text=text, bg=self.bg, fg=self.fg, font=("Segoe UI", 10)); lbl.pack(anchor="w", padx=padx, pady=(pady//2,2)); return lbl
        def add_button(p, text, cmd, color=None, hover=None):
            color = color or self.btn_bg; hover = hover or "#73a0f5"
            b = Button(p, text=text, command=cmd, bg=color, fg=self.btn_fg, relief=FLAT, font=("Segoe UI", 10, "bold"))
            b.pack(fill=X, padx=padx, pady=6); self._hover_button(b, color, hover); return b

        left = Frame(frame, bg=self.bg); left.pack(side=LEFT, fill=Y, padx=(10,6), pady=6)
        right = Frame(frame, bg=self.bg); right.pack(side=LEFT, fill=BOTH, expand=True, padx=(6,10), pady=6)

        add_label_bold(left, "Diffuse Texture")
        add_button(left, "Select Diffuse...", self._apply_pick_diffuse)
        self.apply_diffuse_label = Label(left, text="Loaded: None", bg=self.bg, fg=self.fg); self.apply_diffuse_label.pack(anchor="w", padx=16, pady=(2,6))

        add_label_bold(left, "Lightmap Texture")
        add_button(left, "Select Lightmap...", self._apply_pick_lightmap)
        self.apply_lm_label = Label(left, text="Loaded: None", bg=self.bg, fg=self.fg); self.apply_lm_label.pack(anchor="w", padx=16, pady=(2,6))

        add_label(left, "Mode (Multiply / Add / Mix)")
        mode_menu = OptionMenu(left, self.mode_var, "multiply", "add", "mix"); mode_menu.config(bg=self.entry_bg, fg=self.fg); mode_menu.pack(fill=X, padx=padx, pady=4)

        add_label(left, "Intensity (0 - 100) — real-time")
        intensity_slider = Scale(left, from_=0, to=100, orient=HORIZONTAL, variable=self.intensity_var, bg=self.bg, fg=self.fg, troughcolor=self.entry_bg, highlightthickness=0, command=self._on_intensity_change)
        intensity_slider.set(100); intensity_slider.pack(fill=X, padx=padx, pady=4)
        # Tonemapper selection
        add_label(left, "Tonemapper (preview & LDR save)")
        tonemenu = OptionMenu(left, self.tonemap_choice, "Reinhard", "ACES", "None")
        tonemenu.config(bg=self.entry_bg, fg=self.fg)
        tonemenu.pack(fill=X, padx=padx, pady=4)
        # Tonemap on-save option always enabled by default (UI control removed)
        add_label(left, "Tonemap Strength (0 - 100) — blend between original and tonemapped")
        strength_slider = Scale(left, from_=0, to=100, orient=HORIZONTAL, variable=self.tonemap_strength_var, bg=self.bg, fg=self.fg, troughcolor=self.entry_bg, highlightthickness=0)
        strength_slider.set(100); strength_slider.pack(fill=X, padx=padx, pady=4)
        add_label(left, "Tonemap Exposure (stops) — +/- exposure applied before tonemap")
        exposure_slider = Scale(left, from_=-8, to=8, resolution=0.1, orient=HORIZONTAL, variable=self.tonemap_exposure_var, bg=self.bg, fg=self.fg, troughcolor=self.entry_bg, highlightthickness=0)
        exposure_slider.set(0); exposure_slider.pack(fill=X, padx=padx, pady=4)

        # (Option to treat lightmap as pre-linear removed — handling is automatic now)

        torch_text = "Use PyTorch (GPU if available)" + ("" if TORCH_AVAILABLE else " — (torch not installed)")
        torch_cb = Checkbutton(left, text=torch_text, bg=self.bg, fg=self.fg, variable=self.use_torch_var, onvalue=1, offvalue=0, selectcolor="#000000")
        torch_cb.pack(anchor="w", padx=16, pady=4)

        resize_chk = Checkbutton(left, text="Resize lightmap to diffuse size before composing", bg=self.bg, fg=self.fg, variable=self.resize_var, onvalue=1, offvalue=0, selectcolor="#000000")
        resize_chk.pack(anchor="w", padx=16, pady=2)
        # (Unity multiply checkbox removed)

        add_label_bold(left, "Output Folder")
        add_button(left, "Select Output Folder", self._apply_pick_output_folder)
        self.apply_out_name = Entry(left, bg=self.entry_bg, fg=self.fg); self.apply_out_name.insert(0, "applied_result.png"); self.apply_out_name.pack(fill=X, padx=padx, pady=2)

        add_label(left, "Output Width (px) — leave blank or 0 to keep original")
        self.apply_out_w_var = StringVar(value="0"); w_entry = Entry(left, textvariable=self.apply_out_w_var, bg=self.entry_bg, fg=self.fg); w_entry.pack(fill=X, padx=padx, pady=2)
        add_label(left, "Output Height (px) — leave blank or 0 to keep original")
        self.apply_out_h_var = StringVar(value="0"); h_entry = Entry(left, textvariable=self.apply_out_h_var, bg=self.entry_bg, fg=self.fg); h_entry.pack(fill=X, padx=padx, pady=2)

        btn_row = Frame(left, bg=self.bg); btn_row.pack(fill=X, padx=padx, pady=6)
        preview_btn = Button(btn_row, text="PREVIEW", command=self._do_preview, bg="#f0ad4e", fg=self.btn_fg, relief=FLAT, font=("Segoe UI", 10, "bold")); preview_btn.pack(side=LEFT, expand=True, fill=X, padx=(0,6))
        apply_btn = Button(btn_row, text="APPLY LIGHTMAP", command=self._do_apply, bg="#4CAF50", fg=self.btn_fg, relief=FLAT, font=("Segoe UI", 10, "bold")); apply_btn.pack(side=LEFT, expand=True, fill=X, padx=(6,0))
        self._hover_button(preview_btn, "#f0ad4e", "#f7bf6e"); self._hover_button(apply_btn, "#4CAF50", "#66bb6a")

        # previews on right
        lab = Label(right, text="Previews", bg=self.bg, fg=self.fg, font=("Segoe UI", 11, "bold")); lab.pack(padx=6, pady=6)
        self.preview_diffuse_label = Label(right, bg=self.bg); self.preview_diffuse_label.pack(padx=6, pady=6)
        self.preview_lm_label = Label(right, bg=self.bg); self.preview_lm_label.pack(padx=6, pady=6)
        lab2 = Label(right, text="Result Preview", bg=self.bg, fg=self.fg, font=("Segoe UI", 10, "bold")); lab2.pack(padx=6, pady=(12,2))
        self.preview_result_label = Label(right, bg=self.bg); self.preview_result_label.pack(padx=6, pady=6)

    def _apply_pick_diffuse(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.tga;*.bmp")])
        if p:
            self.diffuse_path = p; self.apply_diffuse_label.config(text=f"Loaded: {Path(p).name}"); self.log(f"[Apply] Selected diffuse: {p}")
            self._update_preview_thumb(self.preview_diffuse_label, p)

    def _apply_pick_lightmap(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.tga;*.bmp")])
        if p:
            self.lm_path = p; self.apply_lm_label.config(text=f"Loaded: {Path(p).name}"); self.log(f"[Apply] Selected lightmap: {p}")
            self._update_preview_thumb(self.preview_lm_label, p)

    def _apply_pick_output_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.apply_output_dir = p; self.log(f"[Apply] Output folder: {p}")

    def _update_preview_thumb(self, label_widget, image_path):
        try:
            is_hdr, data = _load_any_image(Path(image_path))
            if is_hdr:
                # simple tonemap for preview
                arr = data
                rgb = arr[..., :3]
                tm = rgb / (1.0 + rgb)
                srgb = _linear_to_srgb_np(np.clip(tm, 0.0, None))
                a = arr[..., 3:4] if arr.shape[2] > 3 else np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
                rgba = (np.clip(srgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                alpha8 = (np.clip(a, 0.0, 1.0) * 255.0).round().astype(np.uint8)
                img = Image.fromarray(np.concatenate([rgba, alpha8], axis=2), mode='RGBA')
            else:
                img = data if isinstance(data, Image.Image) else Image.open(image_path)
            img.thumbnail((160,160))
            tkimg = ImageTk.PhotoImage(img)
            label_widget.image = tkimg
            label_widget.config(image=tkimg)
        except Exception as e:
            self.log(f"[Preview] thumb failed: {e}")

    def _update_result_preview(self, pil_image):
        try:
            img = pil_image.copy(); img.thumbnail((240,240))
            tkimg = ImageTk.PhotoImage(img)
            self.preview_result_label.image = tkimg
            self.preview_result_label.config(image=tkimg)
        except Exception as e:
            self.log(f"[Preview] result error: {e}")

    # ---------- preview key & debounce ----------
    def _make_preview_key(self, diff, lm, mode, intensity, resize_flag, use_linear, use_torch, unity_mul=False):
        try:
            d = Path(diff); l = Path(lm)
            d_stat = d.stat(); l_stat = l.stat()
            # normalize intensity values so keys compare reliably (support both 0..1 and 0..100 ranges)
            try:
                fi = float(intensity)
                if fi > 1.5:
                    fi = fi / 100.0
                fi = round(fi, 4)
            except Exception:
                fi = float(intensity)
                return (str(l.resolve()), l_stat.st_mtime, l_stat.st_size,
                    str(d.resolve()), d_stat.st_mtime, d_stat.st_size,
                    mode, fi, bool(resize_flag), bool(use_linear), bool(use_torch), bool(unity_mul))
        except Exception:
            try:
                fi = float(intensity)
                if fi > 1.5:
                    fi = fi / 100.0
                fi = round(fi, 4)
            except Exception:
                fi = float(intensity)
            return (lm, diff, mode, fi, bool(resize_flag), bool(use_linear), bool(use_torch), bool(unity_mul))

    def _on_intensity_change(self, val):
        # live real-time slider -> debounce preview so we don't spam heavy recompute
        if self._preview_debounce_after:
            self.root.after_cancel(self._preview_debounce_after)
        self._preview_debounce_after = self.root.after(180, self._do_preview)  # 180 ms debounce

    # ---------- preview & apply ----------
    def _do_preview(self):
        if not self.diffuse_path or not self.lm_path:
            messagebox.showerror("Error", "Select both diffuse and lightmap files.")
            return
        try:
            mode = self.mode_var.get(); intensity = float(self.intensity_var.get())/100.0
            resize_flag = bool(self.resize_var.get())
            use_torch = bool(self.use_torch_var.get()) and TORCH_AVAILABLE
            use_linear = bool(self.use_linear_var.get())
            # Unity multiply option removed; preview follows `use_linear` setting
            # build preview (downscale)
            max_dim = 750  # preview size
            # HDR-aware load -> produce small preview images
            diff_is_hdr = _is_hdr_ext(Path(self.diffuse_path))
            lm_is_hdr = _is_hdr_ext(Path(self.lm_path))

            if diff_is_hdr or lm_is_hdr:
                # load arrays and tonemap to LDR for preview
                diff_hdr, diff_data = _load_any_image(Path(self.diffuse_path))
                lm_hdr, lm_data = _load_any_image(Path(self.lm_path))

                def tonemap_for_preview(arr):
                    try:
                        method = self.tonemap_choice.get()
                    except Exception:
                        method = 'Reinhard'
                    strength = float(self.tonemap_strength_var.get())/100.0 if hasattr(self, 'tonemap_strength_var') else 1.0
                    exposure = float(self.tonemap_exposure_var.get()) if hasattr(self, 'tonemap_exposure_var') else 0.0
                    return _apply_tonemap_to_image(arr, method=method if method != 'None' else 'None', strength=strength, exposure=exposure)

                if diff_is_hdr:
                    d_preview = tonemap_for_preview(diff_data)
                else:
                    d_preview = Image.open(self.diffuse_path).convert('RGBA')
                if lm_is_hdr:
                    l_preview = tonemap_for_preview(lm_data)
                else:
                    l_preview = Image.open(self.lm_path).convert('RGBA')

                d_preview.thumbnail((max_dim, max_dim))
                if resize_flag:
                    l_preview = _resize_rgba_premult(l_preview, d_preview.size)
                else:
                    l_preview.thumbnail((max_dim, max_dim))
                d_img = d_preview
                lm_img = l_preview
            else:
                d_img = Image.open(self.diffuse_path).convert("RGBA")
                lm_img = Image.open(self.lm_path).convert("RGBA")
                d_img.thumbnail((max_dim, max_dim))
                if resize_flag:
                    lm_img = _resize_rgba_premult(lm_img, d_img.size)
                else:
                    lm_img.thumbnail((max_dim, max_dim))

            # Note: option to treat lightmap as pre-linear was removed — handling is automatic.

            base = Image.new("RGBA", d_img.size, (0,0,0,0))
            base.paste(lm_img, (0,0), lm_img)
            layer = Image.new("RGBA", d_img.size, (0,0,0,0))
            layer.paste(d_img, (0,0), d_img)

            # compute key and cache check
            key = self._make_preview_key(self.diffuse_path, self.lm_path, mode, intensity, resize_flag, use_linear, use_torch)
            # always compute preview with fast path (torch/numpy) — it's downscaled so should be quick
            result_img = compose_images(base, layer, mode, intensity, use_linear, use_torch)
            # If user selected a tonemap for preview, apply it (compose_images returns sRGB PIL)
            try:
                method = self.tonemap_choice.get()
            except Exception:
                method = 'Reinhard'
            if method != 'None':
                try:
                    # convert result_img (sRGB PIL) -> linear float arr -> tonemap -> PIL
                    arr = (np.asarray(result_img).astype(np.float32) / 255.0)
                    rgb = arr[..., :3]
                    lin = _srgb_to_linear_np(rgb)
                    # preserve alpha
                    if arr.shape[2] > 3:
                        a = arr[..., 3:4]
                        lin_a = np.concatenate([lin, a], axis=2)
                    else:
                        lin_a = np.concatenate([lin, np.ones((lin.shape[0], lin.shape[1], 1), dtype=np.float32)], axis=2)
                    strength = float(self.tonemap_strength_var.get())/100.0 if hasattr(self, 'tonemap_strength_var') else 1.0
                    exposure = float(self.tonemap_exposure_var.get()) if hasattr(self, 'tonemap_exposure_var') else 0.0
                    result_img = _apply_tonemap_to_image(lin_a, method=method, strength=strength, exposure=exposure)
                except Exception:
                    pass
            self._last_preview_pil = result_img.copy()
            self._last_preview_key = key
            self.log("[Preview] generated")
            self._update_result_preview(result_img)
        except Exception as e:
            messagebox.showerror("Error", str(e)); self.log(f"[Preview] Error: {e}")

    def _do_apply(self):
        if not self.diffuse_path or not self.lm_path:
            messagebox.showerror("Error", "Select both diffuse and lightmap files.")
            return
        # run in background thread to keep UI responsive
        def _apply_job():
            try:
                mode = self.mode_var.get(); intensity = float(self.intensity_var.get())/100.0
                resize_flag = bool(self.resize_var.get())
                use_torch = bool(self.use_torch_var.get()) and TORCH_AVAILABLE
                use_linear = bool(self.use_linear_var.get())
                # resample selection removed; use defaults

                out_w = None; out_h = None
                w_text = self.apply_out_w_var.get().strip(); h_text = self.apply_out_h_var.get().strip()
                if w_text not in ("", "0"): out_w = int(w_text)
                if h_text not in ("", "0"): out_h = int(h_text)

                out_dir = self.apply_output_dir or Path(self.diffuse_path).parent
                out_name = self.apply_out_name.get().strip() or "applied_result.png"
                out_path = Path(out_dir) / out_name

                # check cache: only reuse if cached preview key matches full-res key AND cached preview size == diffuse size
                try:
                    full_key = self._make_preview_key(self.diffuse_path, self.lm_path, mode, intensity, resize_flag, use_linear, use_torch)
                except Exception:
                    full_key = None

                use_cache = False
                if self._last_preview_pil is not None and self._last_preview_key is not None and full_key is not None and full_key == self._last_preview_key:
                    try:
                        diff_size = Image.open(self.diffuse_path).convert("RGBA").size
                        if self._last_preview_pil.size == diff_size:
                            result_img = self._last_preview_pil.copy()
                            use_cache = True
                        else:
                            use_cache = False
                    except Exception:
                        use_cache = False

                if not use_cache:
                    # compose at full-res
                    result_img = compose_files_to_image(Path(self.diffuse_path), Path(self.lm_path),
                                                           mode=mode, intensity=float(intensity),
                                                           resize_lm_to_diffuse=resize_flag,
                                                           use_linear=use_linear,
                                                           diffuse_over_lm=False, use_torch=use_torch,
                                                           srgb_multiply=False)

                # resize and save output (handle HDR numpy array results)
                pil_for_cache = None
                if isinstance(result_img, np.ndarray):
                    # result is linear float32 ndarray
                    if out_w and out_h:
                        # resize via PIL (may clamp HDR values)
                        pil_tmp = Image.fromarray((np.clip(_linear_to_srgb_np(np.clip(result_img[..., :3], 0.0, None)), 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB')
                        pil_a = Image.fromarray((np.clip(result_img[..., 3]*255.0, 0, 255).astype(np.uint8)), mode='L')
                        pil_tmp.putalpha(pil_a)
                        pil_tmp = pil_tmp.resize((out_w, out_h), getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS)
                        result_img = np.asarray(pil_tmp).astype(np.float32) / 255.0
                    # decide how to save: HDR extensions -> save as HDR, otherwise tonemap to LDR if enabled
                    if out_path.suffix.lower() in ('.exr', '.hdr', '.pfm'):
                        _save_any_image(out_path, result_img)
                        # create a tonemapped preview for cache
                        try:
                            strength = float(self.tonemap_strength_var.get())/100.0 if hasattr(self, 'tonemap_strength_var') else 1.0
                            exposure = float(self.tonemap_exposure_var.get()) if hasattr(self, 'tonemap_exposure_var') else 0.0
                            pil_for_cache = _apply_tonemap_to_image(result_img, method=self.tonemap_choice.get(), strength=strength, exposure=exposure)
                        except Exception:
                            pil_for_cache = None
                    else:
                        # LDR output requested
                        strength = float(self.tonemap_strength_var.get())/100.0 if hasattr(self, 'tonemap_strength_var') else 1.0
                        exposure = float(self.tonemap_exposure_var.get()) if hasattr(self, 'tonemap_exposure_var') else 0.0
                        # Always apply tonemap for LDR outputs
                        try:
                            pil_out = _apply_tonemap_to_image(result_img, method=self.tonemap_choice.get(), strength=strength, exposure=exposure)
                        except Exception:
                            # fallback to simple clamp + linear->sRGB
                            rgb = result_img[..., :3]
                            srgb = _linear_to_srgb_np(np.clip(rgb, 0.0, None))
                            a = result_img[..., 3:4] if result_img.shape[2] > 3 else np.ones((result_img.shape[0], result_img.shape[1], 1), dtype=np.float32)
                            rgba = (np.concatenate([srgb, a], axis=2) * 255.0).round().astype(np.uint8)
                            pil_out = Image.fromarray(rgba, mode='RGBA')
                        # resize if requested
                        if out_w and out_h:
                            res = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                            pil_out = pil_out.resize((out_w, out_h), res)
                        pil_out.save(str(out_path))
                        pil_for_cache = pil_out
                else:
                    # PIL Image
                    if out_w and out_h:
                        res = getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                        result_img = result_img.resize((out_w, out_h), res)
                    # If user requested tonemap on save, apply it even for PIL results
                    # Always apply tonemap on save for PIL results; fall back to direct save if tonemap fails
                    try:
                        arr = (np.asarray(result_img).astype(np.float32) / 255.0)
                        rgb = arr[..., :3]
                        lin = _srgb_to_linear_np(rgb)
                        if arr.shape[2] > 3:
                            a = arr[..., 3:4]
                            lin_a = np.concatenate([lin, a], axis=2)
                        else:
                            lin_a = np.concatenate([lin, np.ones((lin.shape[0], lin.shape[1], 1), dtype=np.float32)], axis=2)
                        strength = float(self.tonemap_strength_var.get())/100.0 if hasattr(self, 'tonemap_strength_var') else 1.0
                        exposure = float(self.tonemap_exposure_var.get()) if hasattr(self, 'tonemap_exposure_var') else 0.0
                        pil_out = _apply_tonemap_to_image(lin_a, method=self.tonemap_choice.get(), strength=strength, exposure=exposure)
                        pil_out.save(str(out_path))
                        pil_for_cache = pil_out
                    except Exception:
                        try:
                            _save_any_image(out_path, result_img)
                        except Exception:
                            result_img.save(str(out_path))
                        # convert to PIL preview for cache
                        try:
                            arr = (np.asarray(result_img).astype(np.float32) / 255.0)
                            rgb = arr[..., :3]
                            srgb = _linear_to_srgb_np(np.clip(rgb, 0.0, 1.0))
                            a = arr[..., 3:4] if arr.shape[2] > 3 else np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
                            rgba = (np.concatenate([srgb, a], axis=2) * 255.0).round().astype(np.uint8)
                            pil_for_cache = Image.fromarray(rgba, mode='RGBA')
                        except Exception:
                            pil_for_cache = None
                # update cache: prefer the actual PIL used for preview/save
                try:
                    if pil_for_cache is not None:
                        self._last_preview_pil = pil_for_cache.copy()
                        self._last_preview_key = full_key
                    else:
                        # fallback: cache PIL results if available
                        if not isinstance(result_img, np.ndarray):
                            self._last_preview_pil = result_img.copy()
                            self._last_preview_key = full_key
                except Exception:
                    pass

                # UI update on main thread
                self.root.after(0, lambda: self._on_apply_done(out_path))
            except Exception as e:
                self.root.after(0, lambda: (messagebox.showerror("Error", str(e)), self.log(f"[Apply] Error: {e}")))

        self.log("[Apply] started...")
        self._executor.submit(_apply_job)

    def _on_apply_done(self, out_path):
        self.log(f"[Apply] Saved: {out_path}")
        self._update_result_preview(self._last_preview_pil if self._last_preview_pil is not None else Image.new("RGBA",(128,128),(0,0,0,0)))
        messagebox.showinfo("Success", f"Applied lightmap and saved:\n{out_path}")

    # ---------- shortcuts ----------
    def _setup_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self._crop_pick_lightmap())
        self.root.bind("<Control-d>", lambda e: self._apply_pick_diffuse())
        self.root.bind("<Control-s>", lambda e: self._shortcut_crop_and_apply())

    def _shortcut_crop_and_apply(self):
        try:
            if self.crop_image_path:
                self._do_crop()
            else:
                self.log("[Shortcut] No lightmap selected for crop — skipping crop.")
        except Exception as e:
            self.log(f"[Shortcut] Crop error: {e}")
        try:
            if self.diffuse_path and self.lm_path:
                self._do_apply()
            else:
                self.log("[Shortcut] Diffuse or lightmap missing — skipping apply.")
        except Exception as e:
            self.log(f"[Shortcut] Apply error: {e}")

# run
if __name__ == "__main__":
    root = Tk()
    try:
        icon_file = resource_path("light_map.png")
        if icon_file.exists():
            from tkinter import PhotoImage
            img = PhotoImage(file=str(icon_file)); root.iconphoto(False, img)
    except Exception:
        pass
    app = LightmapApp(root)
    root.mainloop()