from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from itertools import product
import numpy as np
import cv2
import time
from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures

from pathlib import Path
TEXTURE_DIR = Path(__file__).parent / "static" / "textures"
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()

# HSV color ranges for sticker classification
COLOR_RANGES = {
    # White: low saturation, high value
    'W': ((0, 0, 210), (180, 30, 255)),

    # Yellow
    'Y': ((25, 80, 80), (35, 255, 255)),

    # Red (two ranges: 0–10 and 170–180)
    'R1': ((0, 120, 70), (10, 255, 255)),
    'R2': ((170, 120, 70), (180, 255, 255)),

    # Orange (may need to tweak as needed for the rubiks stickers)
    'O': ((10, 120, 70), (22, 255, 255)),

    # Green
    'G': ((45, 60, 60), (85, 255, 255)),

    # Blue
    'B': ((90, 60, 60), (130, 255, 255)),
}

def rotate_grid_cw(grid: List[List[str]], k: int) -> List[List[str]]:
    """Rotate a 3x3 grid k times 90° clockwise."""
    k %= 4
    g = [row[:] for row in grid]
    for _ in range(k):
        g = [[g[2 - c][r] for c in range(3)] for r in range(3)]
    return g

def flatten_grid(grid: List[List[str]]) -> str:
    return ''.join(''.join(row) for row in grid)


def order_points(pts: np.ndarray) -> np.ndarray:
    # sort by x-coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab left-most and right-most
    left = xSorted[:2, :]
    right = xSorted[2:, :]
    # order top-left vs bottom-left
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype="float32")

def classify_color(avg_hsv: Tuple[float,float,float]) -> Tuple[str, float]:
    """Return (color_char, confidence 0..1). Confidence is naive here."""
    for key, (lo, hi) in COLOR_RANGES.items():
        if all(lo[i] <= avg_hsv[i] <= hi[i] for i in range(3)):
            return 'R' if key in ('R1', 'R2') else key
    raise ValueError(f"unclassified hsv={tuple(round(x,1) for x in avg_hsv)}")
    
    

def extract_face_grid(data: bytes) -> Dict[str, Any]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    max_dim = 900
    h, w = img.shape[:2]
    s = max_dim / max(h, w)

    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # 1) Pre‐process
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 2) Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")

    c = max(contours, key=cv2.contourArea)
    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 3) Quad or fallback
    if len(approx) == 4:
        pts = order_points(approx.reshape(4,2))
    else:
        # fallback to min‐area rectangle
        rect = cv2.minAreaRect(c)
        box  = cv2.boxPoints(rect).astype("float32")
        pts  = order_points(box)

    # 4) Perspective warp as before
    tl, tr, br, bl = pts
    W0 = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H0 = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    cap = 600
    scale = min(1.0, cap / max(W0, H0))
    W = max(3, int(W0 * scale))
    H = max(3, int(H0 * scale))


    dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32")
    M   = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(img, M, (W, H))

    # 5) Split & classify facelets (same as you have)
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    grid: List[List[str]] = []
    conf: List[List[float]] = []
    avg_hsv: List[List[Tuple[float,float,float]]] = []

    cellW, cellH = W//3, H//3
    for r in range(3):
        row_c, row_p, row_h = [], [], []
        for c_ in range(3):
            cell = hsv[r*cellH:(r+1)*cellH, c_*cellW:(c_+1)*cellW]
            avg = cv2.mean(cell)[:3]
            try:
                color = classify_color(avg)
            except ValueError as e:
                # Include the cell coordinate to help the UI show what to re-capture
                raise ValueError(f"Unclassified cell at ({r},{c_}): {e}") from None
            row_c.append(color)
            row_p.append(1.0) # naive confidence
            row_h.append(tuple(float(x) for x in avg))
        grid.append(row_c)
        conf.append(row_p)
        avg_hsv.append(row_h)

    center = grid[1][1]  # middle sticker color
    rotation = 0         # (optionally compute later)
    corners = pts.tolist()

    return {
        "grid": grid,
        "center": center,
        "rotation": rotation,
        "corners": corners,
        "conf": conf,
        "avg_hsv": avg_hsv,
    }

class FaceGrid(BaseModel):
    grid: List[List[str]] = Field(..., description="3x3 of color letters: W/Y/R/O/G/B")
    rotation: int = 0

class SolveRequest(BaseModel):
    faces: Dict[str, FaceGrid]  # keys must be U,R,F,D,L,B

@router.post("/scan_face")
async def scan_face(image: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await image.read()
    result = extract_face_grid(data)
    print(f"→ /scan_face processed in {time.perf_counter()-t0:.2f}s, center={result['center']}")
    return result

@router.post("/solve_from_grids")
async def solve_from_grids(req: SolveRequest):
    # 1) Validate keys
    required = ['U','R','F','D','L','B']
    if sorted(req.faces.keys()) != sorted(required):
        raise HTTPException(400, f"faces must include exactly {required}")

# 1) Collect raw color grids (3x3 each)
    raw_grids: Dict[str, List[List[str]]] = {f: req.faces[f].grid for f in required}
    for f, g in raw_grids.items():
        if len(g) != 3 or any(len(row) != 3 for row in g):
            raise HTTPException(400, f"{f} grid must be 3x3")

    # 2) Build color -> face mapping from centers (center is invariant to rotation)
    centers = [raw_grids[f][1][1] for f in required]
    if len(set(centers)) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers}")
    color_to_face = {centers[i]: required[i] for i in range(6)}

    # Optional preflight: verify we only saw these 6 colors 9 times each total
    all_colors = ''.join(flatten_grid(raw_grids[f]) for f in required)
    color_counts = {c: all_colors.count(c) for c in set(all_colors)}
    bad = [c for c in color_counts if c not in color_to_face or color_counts[c] != 9]
    if bad:
        raise HTTPException(400, f"color count issue: {color_counts}. "
                                 f"Expect each of {list(color_to_face.keys())} to appear 9 times.")

    # 3) Try auto-rotations.
    # Anchor U at 0° to cut search space, try all 4 for the other 5 faces (4^5=1024 combos).
    faces_to_search = ['R', 'F', 'D', 'L', 'B']
    for combo in product(range(4), repeat=len(faces_to_search)):
        rot_map = {'U': 0}
        rot_map.update({faces_to_search[i]: combo[i] for i in range(len(faces_to_search))})

        # Build solver string in URFDLB order after rotating each face and mapping colors -> face letters
        mapped_facelets = []
        for f in required:
            g_rot = rotate_grid_cw(raw_grids[f], rot_map[f])
            s_color = flatten_grid(g_rot)
            try:
                s_face = ''.join(color_to_face[ch] for ch in s_color)
            except KeyError as e:
                # an unexpected color slipped in
                raise HTTPException(400, f"unknown color '{e.args[0]}' (centers={centers})")
            mapped_facelets.append(s_face)

        cube_str = ''.join(mapped_facelets)

        # Counts sanity on face letters
        if any(cube_str.count(face) != 9 for face in required):
            continue

        # Try solving; if invalid orientation/permutation, the solver will raise
        try:
            moves = solve_cube(cube_str).split()
            # SUCCESS — generate textures using the same rotations so the viewer matches
            from pathlib import Path
            TEXTURE_DIR = Path(__file__).parent / "static" / "textures"
            TEXTURE_DIR.mkdir(parents=True, exist_ok=True)

            # Rotate color grids the same way for the textures:
            rotated_for_textures = {
                f: flatten_grid(rotate_grid_cw(raw_grids[f], rot_map[f]))
                for f in required
            }
            generate_all_textures(rotated_for_textures, out_dir=str(TEXTURE_DIR))
            textures = {f: f"/static/textures/{f}.png" for f in required}

            return {"solution": moves, "textures": textures, "rotations": rot_map}
        except Exception:
            # try next rotation combo
            continue

    # If we got here, no rotation combo yielded a valid cube
    raise HTTPException(
        400,
        "auto-rotation failed: could not find a consistent orientation. "
        "Please recapture with the on-screen alignment guides."
    )

    # 4) Solve
try:
        moves = solve_cube(cube_str).split()
    except Exception as e:
        raise HTTPException(400, f"solver error: {e}")

    # 5) Generate textures from color strings for 3D viewer
    generate_all_textures(color_facelets, out_dir=str(TEXTURE_DIR))
    host_prefix = ""  # leave empty here; build full URLs in client
    textures = {f: f"{host_prefix}/static/textures/{f}.png" for f in required}

    return {"solution": moves, "textures": textures}




