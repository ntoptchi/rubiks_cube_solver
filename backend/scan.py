from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List, Dict, Tuple
from itertools import product
from pathlib import Path
import time

import numpy as np
import cv2

from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures
from .schemas import SolveRequest

# -------------------- Output dir --------------------
TEXTURE_DIR = Path(__file__).parent / "static" / "textures"
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# -------------------- Color configuration --------------------
# OpenCV HSV: H in [0..180], S,V in [0..255]
# Tweaks: wider blue; white more permissive on V; orange needs higher S.
# -------------------- Color configuration --------------------
# OpenCV HSV: H in [0..180], S,V in [0..255]
# Tweaks for orange detection: narrower white, higher S for orange, small widen on blue.
COLOR_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # White: very low saturation only — prevents bright orange/yellow from becoming white
    'W':  ((0,   0,  190), (180,  30, 255)),  # S <= 30, V >= 190

    # Red (wrap): tight near 0/180
    'R1': ((0,   90,   60), (10, 255, 255)),
    'R2': ((170, 90,   60), (180,255, 255)),

    # Orange: keep it away from yellow by demanding S high; narrow H window
    'O':  ((12, 120,   70), (22, 255, 255)),

    # Yellow: sits above orange and below green
    'Y':  ((24, 100,   90), (38, 255, 255)),

    # Green: classic mid band
    'G':  ((50,  70,   50), (85, 255, 255)),

    # Blue: wide enough but far from orange/yellow
    'B':  ((95,  70,   50), (135,255, 255)),
}

ANCHOR_HSV: Dict[str, Tuple[float, float, float]] = {
    'W': (  0.0,  10.0, 235.0),
    'Y': ( 30.0, 180.0, 220.0),
    'R': (  0.0, 200.0, 200.0),
    'O': ( 20.0, 200.0, 210.0),  # nudge anchor toward classic orange
    'G': ( 60.0, 200.0, 200.0),
    'B': (115.0, 200.0, 210.0),
}

ANCHOR_HUE: Dict[str, float] = {
    'W': 0.0, 'Y': 30.0, 'O': 20.0, 'R': 0.0, 'G': 60.0, 'B': 115.0,
}


# -------------------- Helpers --------------------
def grayworld_awb(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    meanR, meanG, meanB = np.mean(r), np.mean(g), np.mean(b)
    meanR = max(meanR, 1e-6); meanB = max(meanB, 1e-6)
    kr = meanG / meanR; kb = meanG / meanB
    r = np.clip(r * kr, 0, 255).astype(np.uint8)
    b = np.clip(b * kb, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def apply_clahe_v(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v2 = clahe.apply(v)
    return cv2.cvtColor(cv2.merge([h, s, v2]), cv2.COLOR_HSV2BGR)

def order_points(pts: np.ndarray) -> np.ndarray:
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left = x_sorted[:2, :]; right = x_sorted[2:, :]
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

def circ_diff(a: float, b: float) -> float:
    d = (a - b) % 180.0
    if d > 90.0: d -= 180.0
    return d

def shift_hue(avg_hsv: Tuple[float,float,float], delta: float) -> Tuple[float,float,float]:
    H, S, V = avg_hsv
    return ((H - delta) % 180.0, S, V)

def nearest_color_hsv(avg_hsv: Tuple[float,float,float]) -> str:
    H, S, V = avg_hsv
    best, best_d = None, 1e9
    for c, (h0, s0, v0) in ANCHOR_HSV.items():
        hd = min(hue_dist(H, h0), hue_dist(H, 180.0 if c == 'R' else h0))
        d = 2.2*hd + 0.8*abs(S - s0) + 0.6*abs(V - v0)
        if d < best_d:
            best, best_d = c, d
    return best or 'W'

def preprocess_variants(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    def close(x, k=7):
        return cv2.morphologyEx(x, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    edgesA = cv2.Canny(blur, 50, 150); out.append(("canny_50_150", close(edgesA, 7)))
    edgesB = cv2.Canny(blur, 30, 120); out.append(("canny_30_120", close(edgesB, 7)))
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); out.append(("otsu", otsu))
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2); out.append(("adaptive", adap))
    _, s_mask = cv2.threshold(hsv[...,1], 40, 255, cv2.THRESH_BINARY)
    edgesE = cv2.Canny(blur, 40, 120); out.append(("s_mask_canny", close(cv2.bitwise_and(edgesE, s_mask), 5)))
    return out

def find_quad_from_binary(bin_img: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found")
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.015, 0.05):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2))
    rect = cv2.minAreaRect(c); box  = cv2.boxPoints(rect).astype("float32")
    return order_points(box)

# White detector used in a couple places
def _is_white(H: float, S: float, V: float) -> bool:
    """
    Be stricter about calling something white so orange/yellow don't get misread.
    """
    return (S <= 28 and V >= 200) or (S <= 35 and V >= 225)

def classify_strict(avg_hsv: Tuple[float, float, float]) -> str:
    """W/Y/R/O/G/B only; no guessing; no expansion; will raise if no match."""
    H, S, V = avg_hsv

    # Handle white first with a gate so bright low-S pixels don't bleed.
    if COLOR_RANGES['W'][0][1] <= S <= COLOR_RANGES['W'][1][1] and \
       COLOR_RANGES['W'][0][2] <= V <= COLOR_RANGES['W'][1][2]:
        return 'W'

    # Handle red wrap explicitly.
    lo1, hi1 = COLOR_RANGES['R1']
    lo2, hi2 = COLOR_RANGES['R2']
    if (lo1[0] <= H <= hi1[0] and lo1[1] <= S <= hi1[1] and lo1[2] <= V <= hi1[2]) or \
       (lo2[0] <= H <= hi2[0] and lo2[1] <= S <= hi2[1] and lo2[2] <= V <= hi2[2]):
        return 'R'

    # Others (O, Y, G, B)
    for key in ('O', 'Y', 'G', 'B'):
        lo, hi = COLOR_RANGES[key]
        if lo[0] <= H <= hi[0] and lo[1] <= S <= hi[1] and lo[2] <= V <= hi[2]:
            return key

    raise ValueError(f"no_match H={H:.1f} S={S:.1f} V={V:.1f}")

def classify_cell_median(hsv_roi: np.ndarray) -> str:
    """Median HSV in ROI → strict class (no relaxed, no nearest)."""
    Hm = float(np.median(hsv_roi[..., 0]))
    Sm = float(np.median(hsv_roi[..., 1]))
    Vm = float(np.median(hsv_roi[..., 2]))
    return classify_strict((Hm, Sm, Vm))

def classify_relaxed(avg_hsv: Tuple[float,float,float]) -> str:
    H, S, V = avg_hsv

    # White gate (relaxed)
    if (S <= 35 and V >= 200) or (S <= 45 and V >= 230):
        return 'W'

    # Orange/Yellow tie-break:
    # - If S is high, 12..22 => O, 22..30 => Y
    if S >= 95 and 12.0 <= H <= 30.0:
        return 'O' if H < 22.0 else 'Y'

    # Try expanded ranges one more time
    for key, (lo, hi) in COLOR_RANGES.items():
        lo2, hi2 = expand_range(lo, hi, dh=10, ds=50, dv=50)
        if all(lo2[i] <= avg_hsv[i] <= hi2[i] for i in range(3)):
            return 'R' if key in ('R1','R2') else key

    return nearest_color_hsv(avg_hsv)


def lab_kmeans_fallback(avg_bgr_cells: List[np.ndarray]) -> List[str]:
    labs = []
    for bgr in avg_bgr_cells:
        lab = cv2.cvtColor(bgr[None,None,:], cv2.COLOR_BGR2LAB)[0,0,:]
        labs.append(lab.astype(np.float32))
    samples = np.array(labs, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(samples, K=3, bestLabels=None,
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
    ang = np.deg2rad(np.array(h_list) * 2.0)
    s = np.sin(ang).mean(); c = np.cos(ang).mean()
    a = np.arctan2(s, c);  a = a + 2*np.pi if a < 0 else a
    return float(np.rad2deg(a) / 2.0)

# -------------------- Core extraction --------------------
def extract_face_grid_timed(data: bytes):
    """
    Lenient scanner:
      - Decode → quad → warp
      - Classify with strict → relaxed → LAB/KMeans → nearest-color
      - Never raises for color classification (only for decode/quad/ROI).
    Returns (result_dict, timings)
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # --- decode ---
    arr = np.frombuffer(data, np.uint8)
    img0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("Could not decode image")
    timings['decode'] = time.perf_counter() - t0

    # --- mild preproc (non-destructive) ---
    awb = grayworld_awb(img0)
    img = cv2.addWeighted(img0, 0.5, awb, 0.5, 0.0)  # gentle AWB blend
    img = apply_clahe_v(img)                          # boost dark faces a bit

    # --- downscale ---
    h, w = img.shape[:2]
    max_dim = 900
    s = max_dim / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    # --- variants for quad find ---
    t1 = time.perf_counter()
    variants = preprocess_variants(img)
    timings['preproc_gen'] = time.perf_counter() - t1

    # --- quad detection ---
    t2 = time.perf_counter()
    quad, chosen, last_err = None, None, None
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

    # --- warp ---
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

    # --- classify (strict → relaxed → kmeans → nearest) ---
    t4 = time.perf_counter()
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

    cellW, cellH = W // 3, H // 3
    mx_ratio, my_ratio = 0.28, 0.28  # inner margin

    grid: List[List[str]] = [['?' for _ in range(3)] for _ in range(3)]
    conf: List[List[float]] = [[0.0 for _ in range(3)] for _ in range(3)]
    avg_hsv: List[List[List[float]]] = [[[] for _ in range(3)] for _ in range(3)]
    avg_bgr_cells: List[np.ndarray] = []

    # per-cell medians
    for r in range(3):
        for c_ in range(3):
            x0, y0 = c_ * cellW, r * cellH
            mx, my = int(cellW * mx_ratio), int(cellH * my_ratio)
            roi_hsv = hsv[y0+my:y0+cellH-my, x0+mx:x0+cellW-mx]
            roi_bgr = warp[y0+my:y0+cellH-my, x0+mx:x0+cellW-mx]
            if roi_hsv.size == 0:
                raise ValueError("ROI too small; align the face fully inside the guide.")

            Hm = float(np.median(roi_hsv[..., 0]))
            Sm = float(np.median(roi_hsv[..., 1]))
            Vm = float(np.median(roi_hsv[..., 2]))
            avg_hsv[r][c_] = [Hm, Sm, Vm]
            avg_bgr_cells.append(np.median(roi_bgr.reshape(-1, 3), axis=0))

            # strict first
            try:
                ch = classify_strict((Hm, Sm, Vm))
                grid[r][c_] = ch
                conf[r][c_] = 1.0
            except ValueError:
                pass

    # relaxed where still '?'
    for r in range(3):
        for c_ in range(3):
            if grid[r][c_] == '?':
                Hm, Sm, Vm = avg_hsv[r][c_]
                ch = classify_relaxed((Hm, Sm, Vm))
                grid[r][c_] = ch
                conf[r][c_] = max(conf[r][c_], 0.6)

    # LAB/KMeans for any remaining '?'
    if any(grid[r][c_] == '?' for r in range(3) for c_ in range(3)):
        labels = lab_kmeans_fallback([b.astype(np.uint8) for b in avg_bgr_cells])
        k = 0
        for r in range(3):
            for c_ in range(3):
                if grid[r][c_] == '?':
                    grid[r][c_] = labels[k]
                    conf[r][c_] = max(conf[r][c_], 0.5)
                k += 1

    # FINAL safety net: nearest-color (ensures no '?')
    for r in range(3):
        for c_ in range(3):
            if grid[r][c_] == '?':
                Hm, Sm, Vm = avg_hsv[r][c_]
                grid[r][c_] = nearest_color_hsv((Hm, Sm, Vm))
                conf[r][c_] = max(conf[r][c_], 0.4)

    timings['classify'] = time.perf_counter() - t4

    # Gentle center correction if a color heavily dominates
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
        "avg_hsv": avg_hsv,
        "debug_pass": timings.get('quad_pass', 'unknown'),
    }, timings




# -------------------- Utilities for solver --------------------
def rotate_grid_cw(grid: List[List[str]], k: int) -> List[List[str]]:
    k %= 4
    g = [row[:] for row in grid]
    for _ in range(3 if k == 3 else k):
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
            try: return f"{float(x):.3f}"
            except Exception: return str(x)

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

        # sanity
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
    required = ['U', 'R', 'F', 'D', 'L', 'B']
    if sorted(req.faces.keys()) != sorted(required):
        raise HTTPException(400, f"faces must include exactly {required}")

    raw_grids: Dict[str, List[List[str]]] = {f: req.faces[f].grid for f in required}
    for f, g in raw_grids.items():
        if len(g) != 3 or any(len(row) != 3 for row in g):
            raise HTTPException(400, f"{f} grid must be 3x3")

    def face_mode(grid: List[List[str]]) -> str:
        flat = [ch for row in grid for ch in row]
        return max(set(flat), key=flat.count)

    centers_by_capture: Dict[str, str] = {f: face_mode(raw_grids[f]) for f in required}
    centers = list(centers_by_capture.values())
    if len(set(centers)) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers_by_capture}")
    if set(centers) != set('WYROGB'):
        raise HTTPException(400, f"need exactly W,Y,R,O,G,B centers; got {sorted(set(centers))}")

    # color -> scanned grid
    color_to_grid: Dict[str, List[List[str]]] = {centers_by_capture[f]: raw_grids[f] for f in required}

    def candidate_schemes() -> List[Dict[str, str]]:
        """
        Opposites: (W<->Y), (R<->O), (G<->B).
        Try Up in {W, Y}, Front any non-opposite, both orders for Right/Left.
        """
        opts: List[Dict[str, str]] = []
        pairs = {'W': 'Y', 'Y': 'W', 'R': 'O', 'O': 'R', 'G': 'B', 'B': 'G'}
        colors = set(color_to_grid.keys())
        if colors != set('WYROGB'): return opts

        for up in ['W', 'Y']:
            down = pairs[up]
            for front in [c for c in colors if c not in (up, down)]:
                back = pairs[front]
                rem = [c for c in colors if c not in (up, down, front, back)]
                if len(rem) != 2: continue
                opts.append({'U': up, 'D': down, 'F': front, 'B': back, 'R': rem[0], 'L': rem[1]})
                opts.append({'U': up, 'D': down, 'F': front, 'B': back, 'R': rem[1], 'L': rem[0]})
        return opts

    schemes = candidate_schemes()
    if not schemes:
        raise HTTPException(400, "could not infer a valid color scheme from centers; please re-capture with the guides")

    def _flatten(g: List[List[str]]) -> str:
        return ''.join(''.join(row) for row in g)

    def _rot(g: List[List[str]], k: int) -> List[List[str]]:
        k %= 4
        out = [row[:] for row in g]
        for _ in range(k):
            out = [[out[2 - c][r] for c in range(3)] for r in range(3)]
        return out

    for scheme in schemes:
        color_to_face = {col: face for face, col in scheme.items()}
        faces_by_letter: Dict[str, List[List[str]]] = {f: color_to_grid[scheme[f]] for f in required}

        all_colors = ''.join(_flatten(faces_by_letter[f]) for f in required)
        counts = {c: all_colors.count(c) for c in set(all_colors)}
        if set(counts.keys()) - set('WYROGB'): continue
        if any(counts.get(c, 0) != 9 for c in 'WYROGB'): continue

        faces_to_search = ['R', 'F', 'D', 'L', 'B']
        for combo in product(range(4), repeat=len(faces_to_search)):
            rot_map = {'U': 0}
            rot_map.update({faces_to_search[i]: combo[i] for i in range(len(faces_to_search))})

            mapped_facelets: List[str] = []
            ok = True
            for f in required:
                g_rot = _rot(faces_by_letter[f], rot_map[f])
                s_color = _flatten(g_rot)
                try:
                    s_face = ''.join(color_to_face[ch] for ch in s_color)
                except KeyError:
                    ok = False
                    break
                mapped_facelets.append(s_face)

            if not ok: continue

            cube_str = ''.join(mapped_facelets)
            if any(cube_str.count(face) != 9 for face in required):
                continue

            try:
                moves = solve_cube(cube_str).split()
            except Exception:
                continue

            rotated_for_textures = {f: _flatten(_rot(faces_by_letter[f], rot_map[f])) for f in required}
            generate_all_textures(rotated_for_textures, out_dir=str(TEXTURE_DIR))
            textures = {f: f"/static/textures/{f}.png" for f in required}
            return {
                "solution": moves,
                "textures": textures,
                "rotations": rot_map,
                "scheme": scheme,
            }

    raise HTTPException(
        400,
        "auto-rotation failed: no consistent orientation found. "
        f"centers={centers_by_capture}. Try re-capturing with the on-screen guide."
    )
