from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Dict, Any, Tuple
from itertools import product
from pathlib import Path
import time

import numpy as np
import cv2

from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures
from .schemas import SolveRequest

# Where textures will be written
TEXTURE_DIR = Path(__file__).parent / "static" / "textures"
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# --- Color classification ranges (HSV) ---
# Tune these if your stickers/lighting differ (OpenCV Hue is 0..180)
COLOR_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # White: low saturation, high value
    'W':  ((0,   0,   160), (180,  60, 255)),

    # Yellow
    'Y':  ((20,  40,   80), (40,  255, 255)),

    # Red (wraps hue: two ranges)
    'R1': ((0,   40,   60), (12,  255, 255)),
    'R2': ((168, 40,   60), (180, 255, 255)),

    # Orange
    'O':  ((10,  40,   60), (24,  255, 255)),

    # Green
    'G':  ((45,  30,   40), (85,  255, 255)),

    # Blue
    'B':  ((90,  30,   40), (130, 255, 255)),
}

# Anchors for nearest-color fallback (rough midpoints)
COLOR_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    'W': (  0.0,  10.0, 230.0),
    'Y': ( 30.0, 180.0, 200.0),
    'R': (  0.0, 180.0, 180.0),  # special: also treat 180 as red hue wrap
    'O': ( 17.0, 180.0, 200.0),
    'G': ( 60.0, 180.0, 180.0),
    'B': (110.0, 180.0, 180.0),
}

# ---------- Helpers ----------

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as tl, tr, br, bl."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")

def classify_color_strict(avg_hsv: Tuple[float, float, float]) -> str:
    """Return single-letter color or raise if none matches."""
    for key, (lo, hi) in COLOR_RANGES.items():
        if all(lo[i] <= avg_hsv[i] <= hi[i] for i in range(3)):
            return 'R' if key in ('R1', 'R2') else key
    raise ValueError(f"Unclassified cell HSV={tuple(round(x,1) for x in avg_hsv)}")

def expand_range(lo: Tuple[int,int,int], hi: Tuple[int,int,int],
                 pad_h: int = 6, pad_s: int = 25, pad_v: int = 25) -> Tuple[Tuple[int,int,int], Tuple[int,int,int]]:
    lo2 = (max(0, lo[0]-pad_h), max(0, lo[1]-pad_s), max(0, lo[2]-pad_v))
    hi2 = (min(180, hi[0]+pad_h), min(255, hi[1]+pad_s), min(255, hi[2]+pad_v))
    return lo2, hi2

def hue_dist(h1: float, h2: float) -> float:
    """Circular distance on 0..180 hue scale (OpenCV)."""
    d = abs(h1 - h2)
    return min(d, 180 - d)

def nearest_color(avg_hsv: Tuple[float, float, float]) -> str:
    """Pick closest anchor by weighted HSV distance (last-resort fallback)."""
    H, S, V = avg_hsv
    best_c = None
    best_d = 1e9
    for c, (h0, s0, v0) in COLOR_ANCHORS.items():
        # For red, allow wrap around 0/180
        hd = min(hue_dist(H, h0), hue_dist(H, 180.0 if c == 'R' else h0))
        d = 2.0*hd + 1.0*abs(S - s0) + 0.5*abs(V - v0)
        if d < best_d:
            best_d, best_c = d, c
    return best_c or 'W'

def classify_color_relaxed(avg_hsv: Tuple[float, float, float]) -> str:
    """Try expanded ranges before nearest-color."""
    for key, (lo, hi) in COLOR_RANGES.items():
        lo2, hi2 = expand_range(lo, hi)
        if all(lo2[i] <= avg_hsv[i] <= hi2[i] for i in range(3)):
            return 'R' if key in ('R1','R2') else key
    return nearest_color(avg_hsv)

def preprocess_variants(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate multiple binary masks to try quad detection under varied lighting.
    Returns list of (name, binary_image).
    """
    out: List[Tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def morph_close(edges, k=7):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Pass A: Canny 50/150 + close
    edgesA = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    out.append(("canny_50_150", morph_close(edgesA, 7)))

    # Pass B: Canny 30/120 + close
    edgesB = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 120)
    out.append(("canny_30_120", morph_close(edgesB, 7)))

    # Pass C: Otsu
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("otsu", otsu))

    # Pass D: Adaptive threshold
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    out.append(("adaptive", adap))

    # Pass E: Saturation mask + canny (helps dull lighting)
    s = hsv[...,1]
    _, s_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    edgesE = cv2.Canny(blur, 40, 120)
    out.append(("s_mask_canny", morph_close(cv2.bitwise_and(edgesE, s_mask), 5)))

    return out

def find_quad_from_binary(bin_img: np.ndarray) -> np.ndarray:
    """
    Try to locate a 4-point quad from a binary mask. If exact 4 isn't found,
    fall back to minAreaRect.
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)

    # Try several epsilons to encourage a 4-corner approx
    for eps in (0.02, 0.03, 0.015, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))

    # Fallback to min-area rectangle if not an exact quad
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype("float32")
    return order_points(box)

def extract_face_grid_timed(data: bytes):
    """
    Multipass pipeline:
    - Try multiple pre-processing variants for robust quad detection.
    - Two-stage color classification: strict range → relaxed/nearest.
    - Returns (result_dict, timings)
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # decode
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    timings['decode'] = time.perf_counter() - t0

    # downscale
    h, w = img.shape[:2]
    max_dim = 900
    s = max_dim / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # generate variants
    t1 = time.perf_counter()
    variants = preprocess_variants(img)
    timings['preproc_gen'] = time.perf_counter() - t1

    # try to find quad
    quad = None
    chosen_pass = None
    t2 = time.perf_counter()
    last_err = None
    for name, bin_img in variants:
        try:
            quad = find_quad_from_binary(bin_img)
            chosen_pass = name
            break
        except Exception as e:
            last_err = e
            continue
    if quad is None:
        raise ValueError(f"No contours found (all passes). Last error: {last_err}")
    timings['quad'] = time.perf_counter() - t2
    timings['quad_pass'] = chosen_pass or "unknown"

    # warp
    t3 = time.perf_counter()
    tl, tr, br, bl = quad
    W0 = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H0 = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    cap = 600
    scale = min(1.0, cap / max(W0, H0))
    W = max(3, int(W0 * scale))
    H = max(3, int(H0 * scale))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(img, M, (W, H))
    timings['warp'] = time.perf_counter() - t3

    # classify
    t4 = time.perf_counter()
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    grid, conf, avg_hsv = [], [], []
    cellW, cellH = W // 3, H // 3

    # Stage 1: strict classification, collecting failures to retry
    failures: List[Tuple[int,int,Tuple[float,float,float]]] = []
    for r in range(3):
        row_c, row_p, row_h = [], [], []
        for c_ in range(3):
            x0, y0 = c_ * cellW, r * cellH
            # inner ROI to avoid borders
            mx, my = int(cellW * 0.2), int(cellH * 0.2)
            roi = hsv[y0 + my:y0 + cellH - my, x0 + mx:x0 + cellW - mx]
            Hm = float(np.median(roi[..., 0]))
            Sm = float(np.median(roi[..., 1]))
            Vm = float(np.median(roi[..., 2]))
            avg = (Hm, Sm, Vm)
            try:
                color = classify_color_strict(avg)
                row_c.append(color); row_p.append(1.0); row_h.append(avg)
            except ValueError:
                row_c.append('?'); row_p.append(0.0); row_h.append(avg)
                failures.append((r, c_, avg))
        grid.append(row_c); conf.append(row_p); avg_hsv.append(row_h)

    # Stage 2: relaxed / nearest for any failures
    for r, c_, avg in failures:
        color = classify_color_relaxed(avg)
        grid[r][c_] = color
        conf[r][c_] = 0.6  # lower confidence when using relaxed/nearest

    timings['classify'] = time.perf_counter() - t4

    # Final validation: make sure no '?' remain
    if any(grid[r][c] == '?' for r in range(3) for c in range(3)):
        bad = [(r, c, tuple(round(x,1) for x in avg_hsv[r][c]))
               for r in range(3) for c in range(3) if grid[r][c] == '?']
        raise ValueError(f"Unclassified cells remain after multipass: {bad[:3]}")

    return {
        "grid": grid,
        "center": grid[1][1],
        "rotation": 0,
        "corners": quad.tolist(),
        "conf": conf,
        "avg_hsv": avg_hsv,
        "debug_pass": timings.get('quad_pass', 'unknown'),
    }, timings


def rotate_grid_cw(grid: List[List[str]], k: int) -> List[List[str]]:
    """Rotate a 3x3 grid k times 90° clockwise."""
    k %= 4
    g = [row[:] for row in grid]
    for _ in range(k):
        g = [[g[2 - c][r] for c in range(3)] for r in range(3)]
    return g

def flatten_grid(grid: List[List[str]]) -> str:
    return ''.join(''.join(row) for row in grid)

def face_mode(grid: List[List[str]]) -> str:
    """Majority color on a 3x3 face."""
    flat = [ch for row in grid for ch in row]
    return max(set(flat), key=flat.count)

# ---------- Routes ----------

@router.post("/scan_face")
async def scan_face(image: UploadFile = File(...)):
    """
    Accepts a single face photo (multipart field name 'image'),
    returns a 3×3 grid of color letters plus some debug info.
    """
    try:
        t0 = time.perf_counter()
        data = await image.read()
        print(f"[scan_face] bytes={len(data)}")

        # Multipass extract with timings
        res, timings = extract_face_grid_timed(data)
        t1 = time.perf_counter()

        print(
            "[scan_face] pass=%s decode=%.3fs preproc_gen=%.3fs quad=%.3fs warp=%.3fs classify=%.3fs TOTAL=%.3fs center=%s"
            % (
                res.get('debug_pass'),
                timings.get('decode', 0.0),
                timings.get('preproc_gen', 0.0),
                timings.get('quad', 0.0),
                timings.get('warp', 0.0),
                timings.get('classify', 0.0),
                (t1 - t0),
                res.get('center'),
            )
        )

        # Basic shape + allowed letters validation
        grid = res["grid"]
        if len(grid) != 3 or any(len(row) != 3 for row in grid):
            raise HTTPException(400, "Bad grid shape (expected 3x3).")
        allowed = set("WYROGB")
        for r in range(3):
            for c in range(3):
                if grid[r][c] not in allowed:
                    raise HTTPException(400, f"Unknown color '{grid[r][c]}' at ({r},{c}).")

        # Majority-correct the center if it's the odd one out
        flat = [p for row in grid for p in row]
        mcolor = max(set(flat), key=flat.count)
        if flat.count(mcolor) >= 5 and grid[1][1] != mcolor:
            grid[1][1] = mcolor
            res["grid"] = grid
            res["center"] = mcolor

        return res

    except ValueError as ve:
        # 400 so client can prompt re-capture with guidance
        raise HTTPException(400, str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

@router.post("/solve_from_grids")
async def solve_from_grids(req: SolveRequest):
    required = ['U', 'R', 'F', 'D', 'L', 'B']
    if sorted(req.faces.keys()) != sorted(required):
        raise HTTPException(400, f"faces must include exactly {required}")

    # 1) Collect raw grids and validate 3x3
    raw_grids: Dict[str, List[List[str]]] = {f: req.faces[f].grid for f in required}
    for f, g in raw_grids.items():
        if len(g) != 3 or any(len(row) != 3 for row in g):
            raise HTTPException(400, f"{f} grid must be 3x3")

    # 2) Build color -> face mapping from majority centers
    centers_by_face = {f: face_mode(raw_grids[f]) for f in required}
    if len(set(centers_by_face.values())) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers_by_face}")
    color_to_face = {color: face for face, color in centers_by_face.items()}

    # Optional: color counts sanity
    all_colors = ''.join(flatten_grid(raw_grids[f]) for f in required)
    color_counts = {c: all_colors.count(c) for c in set(all_colors)}
    expected = set(color_to_face.keys())
    bad = [c for c in color_counts if c not in expected or color_counts[c] != 9]
    if bad:
        raise HTTPException(
            400,
            f"color count issue: {color_counts}. Expect each of {sorted(expected)} exactly 9 times."
        )

    # 3) Search rotations (anchor U=0°, try 4^5=1024 combos)
    faces_to_search = ['R', 'F', 'D', 'L', 'B']
    for combo in product(range(4), repeat=len(faces_to_search)):
        rot_map = {'U': 0}
        rot_map.update({faces_to_search[i]: combo[i] for i in range(len(faces_to_search))})

        mapped_facelets: List[str] = []
        for f in required:
            g_rot = rotate_grid_cw(raw_grids[f], rot_map[f])
            s_color = flatten_grid(g_rot)
            try:
                s_face = ''.join(color_to_face[ch] for ch in s_color)
            except KeyError as e:
                raise HTTPException(400, f"unknown color '{e.args[0]}' (centers={centers_by_face})")
            mapped_facelets.append(s_face)

        cube_str = ''.join(mapped_facelets)
        if any(cube_str.count(face) != 9 for face in required):
            continue

        # Try solver for this orientation
        try:
            moves = solve_cube(cube_str).split()
        except Exception:
            continue  # try next rotation combo

        # SUCCESS → write textures with same rotation
        rotated_for_textures = {
            f: flatten_grid(rotate_grid_cw(raw_grids[f], rot_map[f]))
            for f in required
        }
        generate_all_textures(rotated_for_textures, out_dir=str(TEXTURE_DIR))
        textures = {f: f"/static/textures/{f}.png" for f in required}
        return {"solution": moves, "textures": textures, "rotations": rot_map}

    raise HTTPException(
        400,
        "auto-rotation failed: no consistent orientation found; please recapture with alignment guides."
    )
