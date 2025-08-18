from pydantic import BaseModel, Field
from typing import List, Dict

class FaceGrid(BaseModel):
    grid: List[List[str]] = Field(..., description="3x3 of W/Y/R/O/G/B")
    rotation: int = 0

class SolveRequest(BaseModel):
    faces: Dict[str, FaceGrid]
