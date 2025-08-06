from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict
import cv2
import numpy as np
from backend.solver.kociemba_solver import solve_cube

router = APIRouter()

# Helper: map sampled BGR color to cube face letter
COLOR_MAP = {
    'white': 'U',
    'red':   'R',
    'green': 'F',
    'yellow':'D',
    'orange':'L',
    'blue':  'B',
}

# Pre-defined BGR centroids for each color (approximate)
COLOR_CENTROIDS = {
    'white':  np.array([255,255,255]),
    'red':    np.array([  0,  0,255]),
    'green':  np.array([  0,255,  0]),
    'yellow': np.array([  0,255,255]),
    'orange': np.array([  0,165,255]),
    'blue':   np.array([255,  0,  0]),
}


def approximate_color(bgr: np.ndarray) -> str:
    """Find nearest color key given BGR sample"""
    min_dist = float('inf')
    best = None
    for name, centroid in COLOR_CENTROIDS.items():
        dist = np.linalg.norm(bgr - centroid)
        if dist < min_dist:
            min_dist = dist
            best = name
    return COLOR_MAP[best]


def extract_facelets(image_bytes: bytes) -> str:
    """
    1. Load image, detect largest square grid via edge/perspective transform
    2. Warp to a top-down view
    3. Divide into 3x3 cells, sample each cell center color
    4. Map to U/R/F/D/L/B string of length 9
    """
    # 1. Decode and preprocess
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2. Find contours, approximate quad, get perspective transform
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                max_area = area
                best_cnt = approx
    if best_cnt is None:
        raise HTTPException(400, "Could not detect cube face grid")

    pts = best_cnt.reshape(4,2)
    # order pts and compute warp
    rect = order_points(pts)
    dst = np.array([[0,0],[300,0],[300,300],[0,300]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (300,300))

    # 3. Sample 3x3 grid
    facelets = []
    step = 300 // 3
    for y in range(3):
        for x in range(3):
            cx = x * step + step//2
            cy = y * step + step//2
            sample = warp[cy-10:cy+10, cx-10:cx+10]
            avg = sample.mean(axis=(0,1))  # BGR average
            facelets.append(approximate_color(avg))
    return ''.join(facelets)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 unordered points, return them in TL, TR, BR, BL order.
    """
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


@router.post("/scan")
async def scan_and_solve(
    up: UploadFile = File(...),
    right: UploadFile = File(...),
    front: UploadFile = File(...),
    down: UploadFile = File(...),
    left: UploadFile = File(...),
    back: UploadFile = File(...),
):
    # Read all six images
    faces_bytes: Dict[str, bytes] = {}
    for key, file in zip(['U','R','F','D','L','B'], [up, right, front, down, left, back]):
        faces_bytes[key] = await file.read()

    # Extract each face string
    face_strings: List[str] = []
    for key in ['U','R','F','D','L','B']:
        face_strings.append(extract_facelets(faces_bytes[key]))

    cubestr = ''.join(face_strings)
    # Solve
    try:
        moves = solve_cube(cubestr).split()
        return {"solution": moves}
    except Exception as e:
        raise HTTPException(400, f"Solver error: {e}")
    
    