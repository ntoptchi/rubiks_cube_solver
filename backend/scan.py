from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Dict, Tuple
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

# -------------------- Color configuration --------------------
# OpenCV HSV: H in [0..180], S,V in [0..255]
# Loosened bands to be more tolerant (esp. blue/green).
COLOR_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # White: very low saturation, high value
    'W':  ((0,   0,  175), (180,  40, 255)),

    # Yellow
    'Y':  ((24,  80,  90), (38,  255, 255)),

    # Red wraps hue at 0/180
    'R1': ((0,   90,  60), (12,  255, 255)),
    'R2': ((168, 90,  60), (180, 255, 255)),

    # Orange (slightly wider)
    'O':  ((8,   100,  60), (23,  255, 255)),

    # Green (wider and lower S/V floor)
    'G':  ((45,  60,  45), (85,  255, 255)),

    # Blue – much wider and lower S/V floor to help in dim light
    'B':  ((95,  65,  55), (135, 255, 255)),
}

# Anchors for nearest-color fallback
ANCHOR_HSV: Dict[str, Tuple[float, float, float]] = {
    'W': (  0.0,  10.0, 235.0),
    'Y': ( 30.0, 180.0, 220.0),
    'R': (  0.0, 200.0, 200.0),
    'O': ( 17.0, 200.0, 210.0),
    'G': ( 60.0, 200.0, 200.0),
    'B': (115.0, 200.0, 210.0),
}

# -------------------- Helpers --------------------

def circ_diff(a: float, b: float) -> float:
    """Smallest signed circular difference on OpenCV hue [0..180]."""
    d = (a - b) % 180.0
    if d > 90.0:
        d -= 180.0
    return d

# Anchor hue (degrees in OpenCV’s 0..180 scale)
ANCHOR_HUE: Dict[str, float] = {
    'W': 0.0,    # irrelevant, S is tiny
    'Y': 30.0,
    'O': 17.0,
    'R': 0.0,    # we also wrap at 180 for red
    'G': 60.0,
    'B': 115.0,
}

def shift_hsv_hue(avg_hsv: Tuple[float,float,float], delta: float) -> Tuple[float,float,float]:
    """Shift hue by -delta (so passing center->anchor delta recenters hues)."""
    H, S, V = avg_hsv
    H2 = (H - delta) % 180.0
    return (H2, S, V)


def grayworld_awb(bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world auto white balance."""
    b, g, r = cv2.split(bgr)
    meanR, meanG, meanB = np.mean(r), np.mean(g), np.mean(b)
    meanR = max(meanR, 1e-6)  # avoid div by zero
    meanB = max(meanB, 1e-6)
    kr = meanG / meanR
    kb = meanG / meanB
    r = np.clip(r * kr, 0, 255).astype(np.uint8)
    b = np.clip(b * kb, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def apply_clahe_v(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on V channel to boost dark faces."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge([h, s, v2])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as tl, tr, br, bl."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")

def expand_range(lo, hi, dh=10, ds=45, dv=45):
    lo2 = (max(0, lo[0]-dh), max(0, lo[1]-ds), max(0, lo[2]-dv))
    hi2 = (min(180, hi[0]+dh), min(255, hi[1]+ds), min(255, hi[2]+dv))
    return lo2, hi2

def hue_dist(h1: float, h2: float) -> float:
    d = abs(h1 - h2)
    return min(d, 180 - d)

def nearest_color_hsv(avg_hsv: Tuple[float,float,float]) -> str:
    """Weighted nearest anchor in HSV space (fallback)."""
    H, S, V = avg_hsv
    best, best_d = None, 1e9
    for c, (h0, s0, v0) in ANCHOR_HSV.items():
        hd = min(hue_dist(H, h0), hue_dist(H, 180.0 if c == 'R' else h0))  # red wrap
        d = 2.2*hd + 0.8*abs(S - s0) + 0.6*abs(V - v0)
        if d < best_d:
            best, best_d = c, d
    return best or 'W'

def preprocess_variants(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Generate several edge/binary variants to find the cube quad."""
    out: List[Tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    def close(x, k=7):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel)

    edgesA = cv2.Canny(blur, 50, 150)
    out.append(("canny_50_150", close(edgesA, 7)))

    edgesB = cv2.Canny(blur, 30, 120)
    out.append(("canny_30_120", close(edgesB, 7)))

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(("otsu", otsu))

    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    out.append(("adaptive", adap))

    s = hsv[...,1]
    _, s_mask = cv2.threshold(s, 40, 255, cv2.THRESH_BINARY)
    edgesE = cv2.Canny(blur, 40, 120)
    out.append(("s_mask_canny", close(cv2.bitwise_and(edgesE, s_mask), 5)))
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

def classify_strict(avg_hsv: Tuple[float,float,float]) -> str:
    H,S,V = avg_hsv

    # Whites that are just desaturated highlights
    if S < 25 and V > 200:
        return 'W'

    # Nudge: if we are in the O/Y boundary (≈15–25°), prefer Orange unless S is extremely high
    if 15.0 <= H <= 25.0 and 70.0 <= S <= 170.0:
        return 'O'

    # Try expanded ranges
    for key, (lo, hi) in COLOR_RANGES.items():
        lo2, hi2 = expand_range(lo, hi, dh=10, ds=40, dv=40)
        if all(lo2[i] <= avg_hsv[i] <= hi2[i] for i in range(3)):
            return 'R' if key in ('R1','R2') else key

    # Fallback to nearest anchor
    return nearest_color_hsv(avg_hsv)

def classify_relaxed(avg_hsv: Tuple[float,float,float]) -> str:
    H, S, V = avg_hsv

    # Whites: very low S but bright
    if S < 28 and V > 200:
        return 'W'

    # In warm light, Y drifts down; in cool light, O can drift up.
    # Use a small hue window + saturation hint to decide.
    if 16 <= H <= 28:
        # Very orange-ish if hue is quite low or saturation is strong
        if H < 22 or (H < 25 and S > 110):
            return 'O'
        else:
            return 'Y'

    # Expanded ranges for all colors
    for key, (lo, hi) in COLOR_RANGES.items():
        lo2, hi2 = expand_range(lo, hi, dh=10, ds=40, dv=40)
        if all(lo2[i] <= avg_hsv[i] <= hi2[i] for i in range(3)):
            return 'R' if key in ('R1','R2') else key

    # Last-resort nearest anchor
    return nearest_color_hsv(avg_hsv)


def lab_kmeans_fallback(avg_bgr_cells: List[np.ndarray]) -> List[str]:
    """Last-resort: cluster the 9 cell means in LAB (K=3) and map to nearest anchors."""
    labs = []
    for bgr in avg_bgr_cells:
        lab = cv2.cvtColor(bgr[None,None,:], cv2.COLOR_BGR2LAB)[0,0,:]
        labs.append(lab.astype(np.float32))
    samples = np.array(labs, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _compactness, labels, centers = cv2.kmeans(samples, K=3, bestLabels=None,
                                               criteria=criteria, attempts=5,
                                               flags=cv2.KMEANS_PP_CENTERS)
    mapped: Dict[int, str] = {}
    for i, c in enumerate(centers):
        lab = c[None,None,:].astype(np.uint8)
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0,0,:]
        hsv = cv2.cvtColor(bgr[None,None,:], cv2.COLOR_BGR2HSV)[0,0,:].astype(float)
        mapped[i] = nearest_color_hsv(tuple(hsv))
    return [mapped[int(k)] for k in labels.flatten()]

def circular_mean_h(h_list: List[float]) -> float:
    """Mean on circular hue (0..180 for OpenCV)."""
    # map to [0..pi], do vector mean, map back to [0..180]
    ang = np.deg2rad(np.array(h_list) * 2.0)  # *2 to map 0..180 -> 0..360 deg
    s = np.sin(ang).mean()
    c = np.cos(ang).mean()
    a = np.arctan2(s, c)
    if a < 0:
        a += 2*np.pi
    return float(np.rad2deg(a) / 2.0)

def extract_face_grid_timed(data: bytes):
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # decode
    arr = np.frombuffer(data, np.uint8)
    img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("Could not decode image")
    timings['decode'] = time.perf_counter() - t0

    # pre: gentle AWB + CLAHE (blend to avoid blue hue shift)
    awb = grayworld_awb(img0)
    img = cv2.addWeighted(img0, 0.5, awb, 0.5, 0)
    img = apply_clahe_v(img)

    # downscale
    h, w = img.shape[:2]
    max_dim = 900
    s = max_dim / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # variants
    t1 = time.perf_counter()
    variants = preprocess_variants(img)
    timings['preproc_gen'] = time.perf_counter() - t1

    # quad
    t2 = time.perf_counter()
    quad, chosen = None, None
    last_err = None
    for name, bin_img in variants:
        try:
            quad = find_quad_from_binary(bin_img)
            chosen = name
            break
        except Exception as e:
            last_err = e
    if quad is None:
        raise ValueError(f"No contours found (all passes). Last={last_err}")
    timings['quad'] = time.perf_counter() - t2
    timings['quad_pass'] = chosen or "unknown"

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

    # --- collect per-cell medians first (so we can do per-face normalization) ---
    cells_hsv: List[List[Tuple[float,float,float]]] = []
    cells_bgr: List[List[np.ndarray]] = []
    grid = []; conf = []

    cellW, cellH = W // 3, H // 3
    mxf, myf = 0.32, 0.32  # slightly deeper ROI (was 0.28)
    for r in range(3):
        row_hsv, row_bgr = [], []
        for c_ in range(3):
            x0, y0 = c_ * cellW, r * cellH
            mx, my = int(cellW * mxf), int(cellH * myf)
            roi_hsv = hsv[y0+my:y0+cellH-my, x0+mx:x0+cellW-mx]
            roi_bgr = warp[y0+my:y0+cellH-my, x0+mx:x0+cellW-mx]
            if roi_hsv.size == 0:
                raise ValueError("ROI too small; align the face in the guide.")
            Hm = float(np.median(roi_hsv[..., 0]))
            Sm = float(np.median(roi_hsv[..., 1]))
            Vm = float(np.median(roi_hsv[..., 2]))
            row_hsv.append((Hm, Sm, Vm))
            row_bgr.append(np.median(roi_bgr.reshape(-1,3), axis=0))
        cells_hsv.append(row_hsv)
        cells_bgr.append(row_bgr)

    # ---- Pass A: quick relaxed classify to detect center color (majority) ----
    prelim = [['?' for _ in range(3)] for _ in range(3)]
    flat_colors = []
    for r in range(3):
        for c_ in range(3):
            prelim[r][c_] = classify_relaxed(cells_hsv[r][c_])
            flat_colors.append(prelim[r][c_])

    # Majority-as-center (more robust if center sticker is slightly off)
    mode_color = max(set(flat_colors), key=flat_colors.count)
    center_color = mode_color

    # Hue-align the whole face so the center sits on its anchor hue
    center_h = cells_hsv[1][1][0]
    anchor_h = ANCHOR_HUE.get(center_color, center_h)
    delta = circ_diff(center_h, anchor_h)  # how much we need to move the face hues

    cells_hsv_shift = [[shift_hsv_hue(cells_hsv[r][c_], delta) for c_ in range(3)] for r in range(3)]

    # ---- Pass B: strict over shifted hues, then relaxed fallback ----
    grid = [['?' for _ in range(3)] for _ in range(3)]
    conf = [[0.0 for _ in range(3)] for _ in range(3)]
    failures: List[Tuple[int,int,Tuple[float,float,float]]] = []

    for r in range(3):
        for c_ in range(3):
            avg = cells_hsv_shift[r][c_]
            try:
                color = classify_strict(avg)
                grid[r][c_] = color
                conf[r][c_] = 1.0
            except ValueError:
                failures.append((r, c_, avg))

    for r, c_, avg in failures:
        color = classify_relaxed(avg)
        grid[r][c_] = color
        conf[r][c_] = 0.6

    # ---- Pass C: snap-to-center for near-center hues (helps O vs Y edge) ----
    # If a cell's shifted hue is close to the (shifted) center hue and is well-saturated, make it center_color.
    Hc, Sc, Vc = cells_hsv_shift[1][1]
    for r in range(3):
        for c_ in range(3):
            if grid[r][c_] == center_color:
                continue
            H, S, V = cells_hsv_shift[r][c_]
            if S > 80 and V > 80:
                if hue_dist(H, Hc) <= 8.0:  # within ~8 degrees in OpenCV hue
                    grid[r][c_] = center_color
                    conf[r][c_] = max(conf[r][c_], 0.7)

    # If any still '?', do LAB/kmeans fallback on the *original* BGR means
    if any(grid[r][c_] == '?' for r in range(3) for c_ in range(3)):
        avg_bgr_flat = [cells_bgr[r][c_] for r in range(3) for c_ in range(3)]
        labels = lab_kmeans_fallback([b.astype(np.uint8) for b in avg_bgr_flat])
        k = 0
        for r in range(3):
            for c_ in range(3):
                if grid[r][c_] == '?':
                    grid[r][c_] = labels[k]
                k += 1

    timings['classify'] = time.perf_counter() - t4

    # ensure no '?'
    if any(grid[r][c_] == '?' for r in range(3) for c_ in range(3)):
        bad = [(r, c_) for r in range(3) for c_ in range(3) if grid[r][c_] == '?']
        raise ValueError(f"Unclassified cells remain at {bad}; recapture with the on-screen guide.")

    # enforce center majority one last time
    flat = [p for row in grid for p in row]
    mode_color = max(set(flat), key=flat.count)
    if flat.count(mode_color) >= 5 and grid[1][1] != mode_color:
        grid[1][1] = mode_color


    return {
        "grid": grid,
        "center": grid[1][1],
        "rotation": 0,
        "corners": quad.tolist(),
        "conf": conf,
        "avg_hsv": [[list(x) for x in avg_hsv[i*3:(i+1)*3]] for i in range(3)],
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

# -------------------- Routes --------------------

@router.post("/scan_face")
async def scan_face(image: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        data = await image.read()
        print(f"[scan_face] bytes={len(data)}")

        res, timings = extract_face_grid_timed(data)
        t1 = time.perf_counter()

        def _fmt(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return str(x)

        print(
            "[scan_face] pass=%s decode=%ss preproc_gen=%ss quad=%ss warp=%ss classify=%ss TOTAL=%ss center=%s"
            % (
                res.get('debug_pass') or timings.get('quad_pass', 'unknown'),
                _fmt(timings.get('decode')),
                _fmt(timings.get('preproc_gen')),
                _fmt(timings.get('quad')),
                _fmt(timings.get('warp')),
                _fmt(timings.get('classify')),
                _fmt(t1 - t0),
                res.get('center'),
            )
        )

        # Basic grid validation
        grid = res["grid"]
        allowed = set("WYROGB")
        for r in range(3):
            for c in range(3):
                if grid[r][c] not in allowed:
                    raise HTTPException(400, f"Unknown color '{grid[r][c]}' at ({r},{c}).")

        return res

    except ValueError as ve:
        raise HTTPException(400, str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

@router.post("/solve_from_grids")
async def solve_from_grids(req: SolveRequest):
    """
    Robust solver:
    - Ignores the labels you used while scanning (U/R/F/D/L/B).
    - Uses the center sticker color of each scanned face to reassign faces.
    - Tries both common schemes:
        * Up=White (or Yellow), Front=Green (or Blue), with the implied Right/Left.
    - Searches rotations only (not permutations), which is now safe because faces are
      placed correctly by color.
    """
    required = ['U', 'R', 'F', 'D', 'L', 'B']
    if sorted(req.faces.keys()) != sorted(required):
        raise HTTPException(400, f"faces must include exactly {required}")

    # 1) Pull raw grids and validate 3x3
    raw_grids: Dict[str, List[List[str]]] = {f: req.faces[f].grid for f in required}
    for f, g in raw_grids.items():
        if len(g) != 3 or any(len(row) != 3 for row in g):
            raise HTTPException(400, f"{f} grid must be 3x3")

    def flatten_grid(grid: List[List[str]]) -> str:
        return ''.join(''.join(row) for row in grid)

    def rotate_grid_cw(grid: List[List[str]], k: int) -> List[List[str]]:
        k %= 4
        g = [row[:] for row in grid]
        for _ in range(k):
            g = [[g[2 - c][r] for c in range(3)] for r in range(3)]
        return g

    def face_mode(grid: List[List[str]]) -> str:
        flat = [ch for row in grid for ch in row]
        return max(set(flat), key=flat.count)

    # 2) Build color->grid map based on centers (most frequent on the face)
    #    This frees users from the exact order they scanned.
    centers_by_capture: Dict[str, str] = {f: face_mode(raw_grids[f]) for f in required}
    # ensure uniqueness and valid colors
    centers = list(centers_by_capture.values())
    if len(set(centers)) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers_by_capture}")
    if set(centers) != set(list("WYROGB")):
        raise HTTPException(400, f"need exactly W,Y,R,O,G,B centers; got {sorted(set(centers))}")

    # Map center color -> the actual 3x3 grid we scanned
    color_to_grid: Dict[str, List[List[str]]] = {}
    for label in required:
        c = centers_by_capture[label]
        color_to_grid[c] = raw_grids[label]

    # 3) Candidate color schemes (common Rubik's layouts).
    # Opposites are fixed: (W<->Y), (R<->O), (B<->G).
    # With Up=W and Front=G -> Right=R, Left=O.
    # With Up=W and Front=B -> Right=O, Left=R.
    # With Up=Y, Right/Left swap accordingly.
    def candidate_schemes() -> List[Dict[str, str]]:
        '''
        Enumerate all physically valid color-face mappings consistent with:
        Opposites: (W<->Y), (R<->O), (G<->B).
        Tries Up in {W, Y}, Front in any non-opposite color,
        and both assignments for Right/Left.
        Produces up to 16 schemes in total.
        '''
    opts: List[Dict[str, str]] = []
    pairs = {'W': 'Y', 'Y': 'W', 'R': 'O', 'O': 'R', 'G': 'B', 'B': 'G'}
    colors = set(color_to_grid.keys())
    if colors != set('WYROGB'):
        return opts  # missing a center color — let caller handle

    for up in ['W', 'Y']:
        down = pairs[up]
        # Front can be any color except Up or its opposite (Down)
        for front in [c for c in colors if c not in (up, down)]:
            back = pairs[front]
            # Remaining two colors become Right/Left in both orders
            rem = [c for c in colors if c not in (up, down, front, back)]
            if len(rem) != 2:
                continue
            opts.append({'U': up, 'D': down, 'F': front, 'B': back, 'R': rem[0], 'L': rem[1]})
            opts.append({'U': up, 'D': down, 'F': front, 'B': back, 'R': rem[1], 'L': rem[0]})
    return opts


    schemes = candidate_schemes()
    if not schemes:
        raise HTTPException(400, "could not infer a valid color scheme from centers; please re-capture with the guides")

    # 4) Try each scheme; within each, try all rotations (anchor U=0°).
    for scheme in schemes:
        # color -> face letter map (inverse of scheme)
        color_to_face = {col: face for face, col in scheme.items()}
        # reassign grids to canonical letters by their center colors
        faces_by_letter: Dict[str, List[List[str]]] = {f: color_to_grid[scheme[f]] for f in required}

        # quick color-count sanity
        all_colors = ''.join(flatten_grid(faces_by_letter[f]) for f in required)
        counts = {c: all_colors.count(c) for c in set(all_colors)}
        # only the 6 colors are allowed, each exactly 9 times
        if set(counts.keys()) - set('WYROGB'):
            # contains an unknown letter; try next scheme
            continue
        if any(counts.get(c, 0) != 9 for c in 'WYROGB'):
            # misclassifications; try next scheme
            continue

        faces_to_search = ['R', 'F', 'D', 'L', 'B']
        for combo in product(range(4), repeat=len(faces_to_search)):
            rot_map = {'U': 0}
            rot_map.update({faces_to_search[i]: combo[i] for i in range(len(faces_to_search))})

            mapped_facelets: List[str] = []
            ok = True
            for f in required:
                g_rot = rotate_grid_cw(faces_by_letter[f], rot_map[f])
                s_color = flatten_grid(g_rot)
                try:
                    s_face = ''.join(color_to_face[ch] for ch in s_color)
                except KeyError:
                    ok = False
                    break
                mapped_facelets.append(s_face)

            if not ok:
                continue

            cube_str = ''.join(mapped_facelets)
            # every face letter must appear exactly 9 times
            if any(cube_str.count(face) != 9 for face in required):
                continue

            # Try solver for this orientation
            try:
                moves = solve_cube(cube_str).split()
            except Exception:
                continue  # try next rotation combo

            # SUCCESS → write textures with the same rotation
            rotated_for_textures = {
                f: flatten_grid(rotate_grid_cw(faces_by_letter[f], rot_map[f]))
                for f in required
            }
            generate_all_textures(rotated_for_textures, out_dir=str(TEXTURE_DIR))
            textures = {f: f"/static/textures/{f}.png" for f in required}
            return {
                "solution": moves,
                "textures": textures,
                "rotations": rot_map,
                "scheme": scheme,  # helpful for debugging / client display
            }

    # If we got here, nothing worked.
    raise HTTPException(
        400,
        "auto-rotation failed: no consistent orientation found.\n"
        f"centers={centers_by_capture}. Try re-capturing with the on-screen guide, "
        "and keep each face flat inside the box."
    )

