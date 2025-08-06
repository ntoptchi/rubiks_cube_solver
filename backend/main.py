# backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
from backend.solver.kociemba_solver import solve_cube
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Mapping from your color initials → Kociemba face letters
COLOR_TO_FACE = {
    "W": "U",  # White → Up
    "R": "R",  # Red   → Right
    "G": "F",  # Green → Front
    "Y": "D",  # Yellow→ Down
    "O": "L",  # Orange→ Left
    "B": "B",  # Blue  → Back
}

VALID_COLORS = set(COLOR_TO_FACE.keys())
VALID_FACES = {"U","R","F","D","L","B"}

class CubeState(BaseModel):
    faces: Dict[str, str] = Field(
        ...,
        description="Keys: U,R,F,D,L,B — values: 9‐char strings of color initials (W,R,G,Y,O,B)"
    )

@app.post("/solve")
def solve(cube: CubeState):
    # 1) Check for all six faces
    missing = VALID_FACES - set(cube.faces.keys())
    if missing:
        raise HTTPException(400, f"Missing faces: {', '.join(sorted(missing))}")

    # 2) Map and validate each face string
    facelets = []
    for face in ["U","R","F","D","L","B"]:
        s = cube.faces[face]
        if len(s) != 9:
            raise HTTPException(400, f"Face '{face}' must have exactly 9 characters")
        # Map colors → solver letters
        mapped = []
        for c in s.upper():
            if c not in VALID_COLORS:
                raise HTTPException(400, f"Invalid color '{c}' on face '{face}'")
            mapped.append(COLOR_TO_FACE[c])
        facelets.append("".join(mapped))

    # 3) Build the cube string in URFDLB order
    cube_str = "".join(facelets)

    # 4) Solve and return
    try:
        moves = solve_cube(cube_str).split()
        return {"solution": moves}
    except Exception as e:
        raise HTTPException(400, f"Solver error: {e}")

@app.get("/")
def read_root():
    return {"status": "up"}

