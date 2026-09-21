from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = PROJECT_ROOT / 'outputs' / 'intermediate' / 'E3_comparison' / 'eval_06_h'
OUTPUT = PROJECT_ROOT / 'outputs' / 'figures' / 'scene06_heavy_pipeline_4panel.png'

PANELS = [
    ('Original', SCENE_DIR / 'original.jpg'),
    ('Bilateral Denoised', SCENE_DIR / 'denoised.png'),
    ('Proposed Edge Map', SCENE_DIR / 'edges.png'),
    ('Ground Truth', SCENE_DIR / 'ground_truth.png'),
]


def get_font(size=28, bold=True):
    candidates = []
    if bold:
        candidates += [
            r'C:\Windows\Fonts\arialbd.ttf',
            r'C:\Windows\Fonts\calibrib.ttf',
            r'C:\Windows\Fonts\segoeuib.ttf',
        ]
    candidates += [
        r'C:\Windows\Fonts\arial.ttf',
        r'C:\Windows\Fonts\calibri.ttf',
        r'C:\Windows\Fonts\segoeui.ttf',
    ]
    for p in candidates:
        path = Path(p)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def main():
    print('=' * 70)
    print('CREATING FINAL CLEAN 4-PANEL FIGURE')
    print('=' * 70)

    for label, path in PANELS:
        print(f'{label}: {path}')
        if not path.exists():
            raise FileNotFoundError(f'Missing input file: {path}')

    # Load all four images.
    src = [Image.open(path).convert('RGB') for _, path in PANELS]

    # All project images should already be standardized to 137 x 253.
    # Make every panel exactly the same display size.
    scale = 3
    panel_w = max(im.width for im in src) * scale
    panel_h = max(im.height for im in src) * scale

    label_h = 72
    gap = 18
    side = 20
    bottom = 20

    canvas_w = side * 2 + panel_w * 4 + gap * 3
    canvas_h = label_h + panel_h + bottom

    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    draw = ImageDraw.Draw(canvas)
    font = get_font(24, bold=True)

    for i, ((label, _), im) in enumerate(zip(PANELS, src)):
        x = side + i * (panel_w + gap)

        # Dedicated label area. There is NO overall title.
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            (x + (panel_w - tw) / 2, (label_h - th) / 2),
            label,
            fill='black',
            font=font,
        )

        # Image area starts below the label area.
        enlarged = im.resize((panel_w, panel_h), Image.Resampling.NEAREST)
        y = label_h
        canvas.paste(enlarged, (x, y))

        # Thin border only around image, not text.
        draw.rectangle(
            [x, y, x + panel_w - 1, y + panel_h - 1],
            outline='black',
            width=2,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT.resolve()
    canvas.save(str(output_path), 'PNG')

    print('=' * 70)
    print('DONE')
    print(f'Saved: {output_path}')
    print(f'Size : {canvas.size[0]} x {canvas.size[1]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
