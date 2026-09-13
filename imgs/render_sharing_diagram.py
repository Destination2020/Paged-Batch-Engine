"""Render this diagram's rect/line/text SVG subset with Pillow and fontconfig.

This is a renderer for the accompanying source, not a general SVG converter.
"""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SCALE = 2
STYLES = {
    'title': {'size': 40, 'bold': True},
    'subtitle': {'size': 22, 'fill': '#526782'},
    'heading': {'size': 26, 'bold': True},
    'body': {'size': 20},
    'small': {'size': 17},
    'white': {'fill': '#ffffff'},
    'muted': {'fill': '#526782'},
}


def main():
    root = ET.parse(ROOT / 'pbe_v4_same_gpu_sharing.svg').getroot()
    canvas = Image.new('RGB', (1536 * SCALE, 1024 * SCALE), 'white')
    draw = ImageDraw.Draw(canvas)
    fonts = {}
    for bold in (False, True):
        fonts[bold] = subprocess.check_output(
            ['fc-match', '-f', '%{file}', 'sans:bold' if bold else 'sans'],
            text=True).strip()

    def render(node, inherited):
        tag = node.tag.rsplit('}', 1)[-1]
        style = dict(inherited)
        for cls in node.get('class', '').split():
            style.update(STYLES.get(cls, {}))
        for attr in ('fill', 'text-anchor'):
            if attr in node.attrib:
                style[attr] = node.get(attr)
        def value(key, default=0):
            return float(node.get(key, default)) * SCALE
        if tag == 'rect':
            x, y = value('x'), value('y')
            draw.rounded_rectangle(
                (x, y, x + value('width'), y + value('height')),
                radius=value('rx'), fill=style.get('fill'),
                outline=node.get('stroke'), width=max(1, int(value('stroke-width', 1))))
        elif tag == 'line':
            draw.line((value('x1'), value('y1'), value('x2'), value('y2')),
                      fill=node.get('stroke'), width=SCALE)
        elif tag == 'text':
            font = ImageFont.truetype(fonts[style.get('bold', False)],
                                      int(style.get('size', 20) * SCALE))
            anchor = {'middle': 'ms', 'end': 'rs'}.get(style.get('text-anchor'), 'ls')
            draw.text((value('x'), value('y')), ''.join(node.itertext()),
                      font=font, fill=style.get('fill', '#10244c'), anchor=anchor)
        elif tag in ('svg', 'g'):
            for child in node:
                render(child, style)
    render(root, {})
    canvas.save(ROOT / 'pbe_v4_same_gpu_sharing.png')


if __name__ == '__main__':
    main()
