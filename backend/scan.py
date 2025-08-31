from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from typing import List, Dict, Any, Tuple
from itertools import product
from pathlib import Path
import time

import numpy as np
import cv2

from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures
from .schemas import FaceGrid, SolveRequest

# Where textures will be written
TEXTURE_DIR = Path(__file__).parent / "static" / "textures"
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# --- Color classification ranges (HSV, OpenCV Hue 0..180) ---
COLOR_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    'W':  ((0,   0,   160), (180,  60, 255)),   # unchanged (white handled also by explicit rule)
    'Y':  ((20,  40,   80), (40,  255, 255)),
    'R1': ((0,   40,   60), (12,  255, 255)),
    'R2': ((168, 40,   60), (180, 255, 255)),
    'O':  ((10,  40,   60), (24,  255, 255)),
    'G':  ((45,  30,   40), (95,  255, 255)),   # ↑ upper bound to 95
    'B':  ((95,  25,   35), (140, 255, 255)),   # ↑ widen & allow lower S/V
}

# Anchors for nearest-color fallback
COLOR_ANCHORS: Dict[str, Tuple[float, float, float]] = {
    'W': (  0.0,  10.0, 230.0),
    'Y': ( 30.0, 180.0, 200.0),
    'R': (  0.0, 180.0, 180.0),
    'O': ( 17.0, 180.0, 200.0),
    'G': ( 60.0, 180.0, 180.0),
    'B': (110.0, 180.0, 180.0),
}

def order_points(pts: np.ndarray) -> np.ndarray:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")

def classify_color_strict(avg_hsv: Tuple[float, float, float]) -> str:
    H, S, V = avg_hsv

    # Explicit white rule first: low saturation + high value
    if S <= 30 and V >= 180:
        return 'W'

    # Normal ranges
    for key, (lo, hi) in COLOR_RANGES.items():
        if all(lo[i] <= avg_hsv[i] <= hi[i] for i in range(3)):
            return 'R' if key in ('R1', 'R2') else key

    # Blue/Green boundary nudge (90–100° hue is tricky)
    if 90 <= H <= 100:
        # a slight preference towards Blue if saturation is decent
        return 'B' if S >= 60 else 'G'

    raise ValueError(f"Unclassified cell HSV={tuple(round(x,1) for x in avg_hsv)}")


def expand_range(lo, hi, pad_h=6, pad_s=25, pad_v=25):
    lo2 = (max(0, lo[0]-pad_h), max(0, lo[1]-pad_s), max(0, lo[2]-pad_v))
    hi2 = (min(180, hi[0]+pad_h), min(255, hi[1]+pad_s), min(255, hi[2]+pad_v))
    return lo2, hi2

def hue_dist(h1: float, h2: float) -> float:
    d = abs(h1 - h2)
    return min(d, 180 - d)

def nearest_color(avg_hsv: Tuple[float, float, float]) -> str:
    H, S, V = avg_hsv
    best_c, best_d = None, 1e9
    for c, (h0, s0, v0) in COLOR_ANCHORS.items():
        hd = min(hue_dist(H, h0), hue_dist(H, 180.0 if c == 'R' else h0))
        d = 2.0*hd + 1.0*abs(S - s0) + 0.5*abs(V - v0)
        if d < best_d:
            best_d, best_c = d, c
    return best_c or 'W'

def classify_color_relaxed(avg_hsv: Tuple[float, float, float]) -> str:
    for key, (lo, hi) in COLOR_RANGES.items():
        lo2, hi2 = expand_range(lo, hi)
        if all(lo2[i] <= avg_hsv[i] <= hi2[i] for i in range(3)):
            return 'R' if key in ('R1','R2') else key
    return nearest_color(avg_hsv)

def preprocess_variants(img_bgr: np.ndarray):
    out = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    def morph_close(edges, k=7):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    edgesA = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    out.append(("canny_50_150", morph_close(edgesA, 7)))

    edgesB = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 30, 120)
    out.append(("canny_30_120", morph_close(edgesB, 7)))

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("otsu", otsu))

    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    out.append(("adaptive", adap))

    s = hsv[...,1]
    _, s_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    edgesE = cv2.Canny(blur, 40, 120)
    out.append(("s_mask_canny", morph_close(cv2.bitwise_and(edgesE, s_mask), 5)))
    return out

def find_quad_from_binary(bin_img: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.015, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(c)
    box  = cv2.boxPoints(rect).astype("float32")
    return order_points(box)

def extract_face_grid_timed(data: bytes):
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    timings['decode'] = time.perf_counter() - t0

    h, w = img.shape[:2]
    s = 900 / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

    t1 = time.perf_counter()
    variants = preprocess_variants(img)
    timings['preproc_gen'] = time.perf_counter() - t1

    quad, chosen_pass, last_err = None, None, None
    t2 = time.perf_counter()
    for name, bin_img in variants:
        try:
            quad = find_quad_from_binary(bin_img)
            chosen_pass = name
            break
        except Exception as e:
            last_err = e
    if quad is None:
        raise ValueError(f"No contours found (all passes). Last error: {last_err}")
    timings['quad'] = time.perf_counter() - t2
    timings['quad_pass'] = chosen_pass or "unknown"

    t3 = time.perf_counter()
    tl, tr, br, bl = quad
    W0 = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H0 = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    cap = 600
    scale = min(1.0, cap / max(W0, H0))
    W, H = max(3, int(W0*scale)), max(3, int(H0*scale))
    dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32")
    M   = cv2.getPerspectiveTransform(quad, dst)
    warp = cv2.warpPerspective(img, M, (W, H))
    timings['warp'] = time.perf_counter() - t3

    t4 = time.perf_counter()
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    grid, conf, avg_hsv = [], [], []
    cellW, cellH = W//3, H//3
    failures = []
    for r in range(3):
        row_c, row_p, row_h = [], [], []
        for c_ in range(3):
            x0, y0 = c_*cellW, r*cellH
            mx, my = int(cellW*0.2), int(cellH*0.2)
            roi = hsv[y0+my:y0+cellH-my, x0+mx:x0+cellW-mx]
            Hm = float(np.median(roi[...,0]))
            Sm = float(np.median(roi[...,1]))
            Vm = float(np.median(roi[...,2]))
            avg = (Hm, Sm, Vm)
            try:
                color = classify_color_strict(avg)
                row_c.append(color); row_p.append(1.0); row_h.append(avg)
            except ValueError:
                row_c.append('?'); row_p.append(0.0); row_h.append(avg)
                failures.append((r, c_, avg))
        grid.append(row_c); conf.append(row_p); avg_hsv.append(row_h)

    for r, c_, avg in failures:
        color = classify_color_relaxed(avg)
        grid[r][c_] = color
        conf[r][c_] = 0.6

    timings['classify'] = time.perf_counter() - t4

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
    k %= 4
    g = [row[:] for row in grid]
    for _ in range(k):
        g = [[g[2 - c][r] for c in range(3)] for r in range(3)]
    return g

def flatten_grid(grid: List[List[str]]) -> str:
    return ''.join(''.join(row) for row in grid)

@router.post("/scan_face")
async def scan_face(image: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        data = await image.read()
        res, timings = extract_face_grid_timed(data)
        t1 = time.perf_counter()

        print("[scan_face] pass=%s decode=%.3fs preproc_gen=%.3fs quad=%.3fs warp=%.3fs classify=%.3fs TOTAL=%.3fs center=%s" %
              (res.get('debug_pass'), timings.get('decode',0.0), timings.get('preproc_gen',0.0),
               timings.get('quad',0.0), timings.get('warp',0.0), timings.get('classify',0.0),
               (t1 - t0), res.get('center')))

        # ensure 3x3 letters and nudge center to face majority if off by 1
        grid = res["grid"]
        if len(grid)!=3 or any(len(row)!=3 for row in grid):
            raise HTTPException(400, "Bad grid shape (expected 3x3).")
        flat = [p for row in grid for p in row]
        mode_color = max(set(flat), key=flat.count)
        if flat.count(mode_color) >= 5 and grid[1][1] != mode_color:
            grid[1][1] = mode_color
            res["grid"] = grid
            res["center"] = mode_color

        return res

    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

@router.post("/solve_from_grids")
async def solve_from_grids(req: SolveRequest, request: Request):
    required = ['U', 'R', 'F', 'D', 'L', 'B']
    if sorted(req.faces.keys()) != sorted(required):
        raise HTTPException(400, f"faces must include exactly {required}")

    raw_grids: Dict[str, List[List[str]]] = {f: req.faces[f].grid for f in required}
    for f, g in raw_grids.items():
        if len(g) != 3 or any(len(row) != 3 for row in g):
            raise HTTPException(400, f"{f} grid must be 3x3")

    # center = face majority
    def face_mode(grid: List[List[str]]) -> str:
        flat = [ch for row in grid for ch in row]
        return max(set(flat), key=flat.count)

    centers_by_face = {f: face_mode(raw_grids[f]) for f in required}
    if len(set(centers_by_face.values())) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers_by_face}")

    color_to_face = {color: face for face, color in centers_by_face.items()}

    # sanity on color counts
    all_colors = ''.join(flatten_grid(raw_grids[f]) for f in required)
    color_counts = {c: all_colors.count(c) for c in set(all_colors)}
    expected = set(color_to_face.keys())
    bad = [c for c in color_counts if c not in expected or color_counts[c] != 9]
    if bad:
        raise HTTPException(
            400,
            f"color count issue: {color_counts}. Expect each of {sorted(expected)} exactly 9 times."
        )

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

        try:
            moves = solve_cube(cube_str).split()
        except Exception:
            continue

        rotated_for_textures = {
            f: flatten_grid(rotate_grid_cw(raw_grids[f], rot_map[f]))
            for f in required
        }
        generate_all_textures(rotated_for_textures, out_dir=str(TEXTURE_DIR))
        base = str(request.base_url).rstrip('/')
        textures = {f: f"{base}/static/textures/{f}.png" for f in ['U','R','F','D','L','B']}
        return {"solution": moves, "textures": textures, "rotations": rot_map}

    raise HTTPException(
        400,
        "auto-rotation failed: no consistent orientation found; please recapture with alignment guides."
    )
