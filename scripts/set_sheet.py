"""把某个 SKU 的整套图拼成一张 4 列一览，便于一眼验收。"""
import sys, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
F = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)

def build(sku: str, h: int = 330) -> Path:
    d = ROOT / 'assets' / 'ecom_set' / sku
    fs = [f for f in d.glob('*.png') if '一览' not in f.name]
    order = {'M': 0, 'S': 1, 'D': 2}
    fs.sort(key=lambda p: (order[re.search(r'([MSD])\d', p.name).group(1)],
                           int(re.search(r'\d', p.name).group())))
    tiles = []
    for f in fs:
        im = Image.open(f).convert('RGB')
        tiles.append((f.stem, im.resize((int(im.width * h / im.height), h), Image.LANCZOS)))
    cols, PAD, LAB = 4, 10, 26
    rows = (len(tiles) + cols - 1) // cols
    cw = max(t[1].width for t in tiles)
    cv = Image.new('RGB', (PAD + cols * (cw + PAD), PAD + rows * (h + LAB + PAD)), (248, 248, 246))
    dr = ImageDraw.Draw(cv)
    for i, (name, im) in enumerate(tiles):
        r, c = divmod(i, cols)
        x, y = PAD + c * (cw + PAD), PAD + r * (h + LAB + PAD)
        cv.paste(im, (x, y))
        dr.text((x + 2, y + h + 4), name, font=F, fill=(40, 40, 40))
    out = d / '整套一览.png'
    cv.save(out)
    print(f"{sku}: {len(tiles)} 张 -> {out.relative_to(ROOT)}  {cv.size}")
    return out

if __name__ == '__main__':
    for a in sys.argv[1:]:
        build(a)
