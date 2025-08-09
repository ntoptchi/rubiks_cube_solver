# backend/scan.py

from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
import cv2
from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures

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

def extract_facelets_image(data: bytes) -> str:
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
    face_str, cellW, cellH = "", W//3, H//3
    for r in range(3):
        for c in range(3):
            cell = hsv[r*cellH:(r+1)*cellH, c*cellW:(c+1)*cellW]
            avg = cv2.mean(cell)[:3]
            for key, (lower, upper) in COLOR_RANGES.items():
                if all(lower[i] <= avg[i] <= upper[i] for i in range(3)):
                    color = 'R' if key in ('R1','R2') else key
                    face_str += color
                    break
            else:
                face_str += 'W'  # fallback (you can also raise to force re-capture)

                        
            return face_str

@router.post("/scan")
async def scan_faces(
    up: UploadFile = File(...),
    right: UploadFile = File(...),
    front: UploadFile = File(...),
    down: UploadFile = File(...),
    left: UploadFile = File(...),
    back: UploadFile = File(...),
):
    print("→ /scan called with files:", up.filename, right.filename, front.filename,
          down.filename, left.filename, back.filename)
    try:
        # read all six files
        files = await up.read(), await right.read(), await front.read(), \
                await down.read(), await left.read(), await back.read()

        # extract each into a 9‐char string
        facelets = [extract_facelets_image(data) for data in files]

        print("DEBUG facelets:", facelets)
        cube_str = "".join(facelets)
        print("DEBUG cube_str:", cube_str, "len=", len(cube_str))


        # ─── NEW: generate per‐face textures ───
        face_strs = dict(zip(['U','R','F','D','L','B'], facelets))
        # this writes: backend/static/textures/U.png, etc.
        generate_all_textures(face_strs, out_dir="static/textures")

        # assemble in Kociemba order: U R F D L B
        cube_str = "".join(facelets)
        moves = solve_cube(cube_str).split()
        return {"solution": moves}

    except ValueError as ve:
        raise HTTPException(400, f"Scan error: {ve}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")





