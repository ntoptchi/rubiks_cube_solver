from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path


from .scan import router as scan_router

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Rubik Solver")

# static files (for generated textures)
(static_dir := BASE_DIR / "static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# simple health/root
@app.get("/")
def root():
    return {"ok": True, "service": "rubik-solver"}

@app.get("/health")
def health():
    return {"status": "ok"}

# include the scan router with a single, consistent prefix
app.include_router(scan_router, prefix="/api")

# optional CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
