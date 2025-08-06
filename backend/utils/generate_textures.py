import os
from PIL import Image, ImageDraw

# Map your face‐letter initials to RGB colors
COLOR_MAP = {
    'W': (255, 255, 255),   # White
    'R': (255,   0,   0),   # Red
    'G': (  0, 255,   0),   # Green
    'Y': (255, 255,   0),   # Yellow
    'O': (255, 165,   0),   # Orange
    'B': (  0,   0, 255),   # Blue
}

# Order your faces exactly as you pass them to the solver:
FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']

def make_face_texture(facelets: str, size: int = 512) -> Image.Image:
    """
    facelets: a 9‐char string like "WRGYBO..." in row‐major order
    size: output image is size×size pixels
    returns a PIL Image where each cell is filled with the mapped color
    """
    cell = size // 3
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)
    for idx, letter in enumerate(facelets):
        rgb = COLOR_MAP.get(letter, (128, 128, 128))
        x = (idx % 3) * cell
        y = (idx // 3) * cell
        draw.rectangle([x, y, x + cell, y + cell], fill=rgb)
    return img

def generate_all_textures(face_strs: dict, out_dir: str = "textures"):
    """
    face_strs: dict mapping each of 'U','R','F','D','L','B' to its 9-char string
    Creates textures/U.png, textures/R.png, etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    for face in FACE_ORDER:
        s = face_strs.get(face)
        if not s or len(s) != 9:
            raise ValueError(f"face '{face}' must be a 9‐char string")
        img = make_face_texture(s, size=512)
        path = os.path.join(out_dir, f"{face}.png")
        img.save(path)
        print(f"Saved {path}")

if __name__ == "__main__":
    # Example usage: replace these with the strings you got from extract_facelets
    sample = {
        'U': "WWWWWWWWW",
        'R': "RRRRRRRRR",
        'F': "GGGGGGGGG",
        'D': "YYYYYYYYY",
        'L': "OOOOOOOOO",
        'B': "BBBBBBBBB",
    }
    generate_all_textures(sample)
