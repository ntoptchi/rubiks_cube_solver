# backend/scan.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import numpy as np
import cv2
from io import BytesIO
from solver.kociemba_solver import solve_cube
from utils.generate_textures import generate_all_textures

app = FastAPI()

# HSV color ranges for sticker classification
COLOR_RANGES = {
    'W': ((0, 0, 200), (180, 40, 255)),
    'Y': ((20, 100, 100), (30, 255, 255)),
    'R': ((0, 100, 100), (10, 255, 255)),
    'O': ((10, 100, 100), (20, 255, 255)),
    'G': ((50, 100, 100), (70, 255, 255)),
    'B': ((90, 100, 100), (130, 255, 255)),
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
    # decode image
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # find largest contour (presumed cube face)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found")
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        raise ValueError("Could not detect quadrilateral")

    pts = order_points(approx.reshape(4, 2))
    # compute max width/height
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    # perspective transform
    dst = np.array([[0, 0], [maxW, 0], [maxW, maxH], [0, maxH]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warp = cv2.warpPerspective(img, M, (maxW, maxH))

    # split into 3×3 grid and classify each cell
    face_str = ""
    cellW, cellH = maxW // 3, maxH // 3
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    for row in range(3):
        for col in range(3):
            x0, y0 = col * cellW, row * cellH
            cell = hsv[y0:y0 + cellH, x0:x0 + cellW]
            avg = cv2.mean(cell)[:3]  # average HSV
            # find which color range contains this avg
            for color, (lower, upper) in COLOR_RANGES.items():
                if all(lower[i] <= avg[i] <= upper[i] for i in range(3)):
                    face_str += color
                    break
            else:
                # fallback or error
                face_str += 'W'
    return face_str  # e.g. "WWGRROO...B"

@app.post("/scan")
async def scan_faces(
    up: UploadFile = File(...),
    right: UploadFile = File(...),
    front: UploadFile = File(...),
    down: UploadFile = File(...),
    left: UploadFile = File(...),
    back: UploadFile = File(...),
):
    try:
        # read all six files
        files = await up.read(), await right.read(), await front.read(), \
                await down.read(), await left.read(), await back.read()

        # extract each into a 9‐char string
        facelets = [extract_facelets_image(data) for data in files]

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

from utils.generate_textures import generate_all_textures



