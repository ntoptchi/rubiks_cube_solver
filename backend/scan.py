from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from itertools import product
import numpy as np
import cv2
import time
from .solver.kociemba_solver import solve_cube
from .utils.generate_textures import generate_all_textures
from .schemas import FaceGrid, SolveRequest


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

    # 2) Build color -> face mapping from centers
    centers = [raw_grids[f][1][1] for f in required]
    if len(set(centers)) != 6:
        raise HTTPException(400, f"centers not unique; centers={centers}")
    color_to_face = {centers[i]: required[i] for i in range(6)}

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
                raise HTTPException(400, f"unknown color '{e.args[0]}' (centers={centers})")
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
        texture_dir = Path(__file__).parent / "static" / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        rotated_for_textures = {
            f: flatten_grid(rotate_grid_cw(raw_grids[f], rot_map[f]))
            for f in required
        }
        generate_all_textures(rotated_for_textures, out_dir=str(texture_dir))
        textures = {f: f"/static/textures/{f}.png" for f in required}
        return {"solution": moves, "textures": textures, "rotations": rot_map}

    raise HTTPException(
        400,
        "auto-rotation failed: no consistent orientation found; please recapture with alignment guides."
    )