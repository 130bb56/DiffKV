"""Render publication pie charts from the measured mean E2E percentages.

Uses the system Cairo library through Python's standard-library ctypes.
Run from any directory: python /absolute/path/to/plot.py
"""
import ctypes as ct
import ctypes.util
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'logs/performance/nsys-repeat3-20261018-173725/stats.json'
CASE = 'b16-isl1024-osl1024'
COLORS = ['#df625c', '#5c92c4', '#c4c4c4']
NAMES = ['GEMM', 'Attention', 'others']
LIB = ct.CDLL(ctypes.util.find_library('cairo'))
PTR, DOUBLE, INT, CHAR = ct.c_void_p, ct.c_double, ct.c_int, ct.c_char_p


def bind(name, args, result=None):
    fn = getattr(LIB, 'cairo_' + name)
    fn.argtypes = args
    fn.restype = result
    return fn


image_surface = bind('image_surface_create', [INT, INT, INT], PTR)
pdf_surface = bind('pdf_surface_create', [CHAR, DOUBLE, DOUBLE], PTR)
svg_surface = bind('svg_surface_create', [CHAR, DOUBLE, DOUBLE], PTR)
create = bind('create', [PTR], PTR)
destroy = bind('destroy', [PTR])
finish = bind('surface_finish', [PTR])
surface_destroy = bind('surface_destroy', [PTR])
write_png = bind('surface_write_to_png', [PTR, CHAR], INT)
status = bind('status', [PTR], INT)
scale = bind('scale', [PTR, DOUBLE, DOUBLE])
set_color = bind('set_source_rgb', [PTR, DOUBLE, DOUBLE, DOUBLE])
paint = bind('paint', [PTR])
new_path = bind('new_path', [PTR])
move = bind('move_to', [PTR, DOUBLE, DOUBLE])
arc = bind('arc', [PTR, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE])
close = bind('close_path', [PTR])
fill = bind('fill', [PTR])
font = bind('select_font_face', [PTR, CHAR, INT, INT])
font_size = bind('set_font_size', [PTR, DOUBLE])
show = bind('show_text', [PTR, CHAR])
text_extents = bind('text_extents', [PTR, CHAR, ct.POINTER(DOUBLE)])


def color(ctx, value):
    set_color(ctx, *(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5)))


def text(ctx, label, x, baseline, size=11.5):
    font_size(ctx, size)
    encoded = label.encode('utf-8')
    ext = (DOUBLE * 6)()
    text_extents(ctx, encoded, ext)
    set_color(ctx, 0.06, 0.06, 0.06)
    move(ctx, x - ext[0] - ext[2] / 2, baseline)
    show(ctx, encoded)


def pie(ctx, values, cx, cy, radius):
    angle = -math.pi / 2
    for index, (name, value, shade) in enumerate(zip(NAMES, values, COLORS)):
        end = angle + value / 100 * 2 * math.pi
        new_path(ctx)
        move(ctx, cx, cy)
        arc(ctx, cx, cy, radius, angle, end)
        close(ctx)
        color(ctx, shade)
        fill(ctx)
        middle = (angle + end) / 2
        distance = radius * (0.50 if index == 0 else 0.64)
        x = cx + distance * math.cos(middle)
        y = cy + distance * math.sin(middle)
        text(ctx, name, x, y - 2)
        text(ctx, f'{value:.2f}%', x, y + 12)
        angle = end


def render(stem, series, combined=False):
    width, height = (468, 246) if combined else (234, 226)
    for extension in ('svg', 'pdf', 'png'):
        path = OUT / f'{stem}.{extension}'
        resolution = 600 / 72 if extension == 'png' else 1
        if extension == 'png':
            surface = image_surface(0, round(width * resolution), round(height * resolution))
        else:
            make_surface = pdf_surface if extension == 'pdf' else svg_surface
            surface = make_surface(str(path).encode(), width, height)
        ctx = create(surface)
        scale(ctx, resolution, resolution)
        set_color(ctx, 1, 1, 1)
        paint(ctx)
        font(ctx, b'Liberation Serif', 0, 0)
        if combined:
            text(ctx, 'Batch 16, input length 1024, output length 1024', width / 2, 15, 11)
        for i, (label, values) in enumerate(series):
            cx = 117 + 234 * i
            pie(ctx, values, cx, 116 if combined else 104, 88)
            title = f'({chr(97 + i)}) {label}' if combined else label
            text(ctx, title, cx, 228 if combined else 216, 13)
        assert status(ctx) == 0, f'Cairo rendering error: {path}'
        if extension == 'png':
            assert write_png(surface, str(path).encode()) == 0
        destroy(ctx)
        finish(surface)
        surface_destroy(surface)
        print(path)


def main():
    stats = json.loads(SOURCE.read_text())
    case = next(c for c in stats['cases'] if c['case']['id'] == CASE)
    series = []
    metadata = {'case': CASE, 'source': str(SOURCE),
                'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'statistic': 'mean of per-repetition E2E percentages, n=3',
                'others': '100 - GEMM - Attention, including all remaining E2E time',
                'colors': dict(zip(NAMES, COLORS)), 'png_dpi': 600, 'series': {}}
    for backend, label in [('fp16', 'FP16'), ('diffkv', 'DiffKV v1')]:
        metrics = case['backends'][backend]['metrics']
        values = [metrics[key]['mean_whole_wall_pct'] for key in ('gemm', 'attention')]
        values.append(100 - sum(values))
        assert min(values) >= 0 and abs(sum(values) - 100) < 1e-9
        metadata['series'][backend] = dict(zip(NAMES, values))
        series.append((label, values))
        render(backend, [(label, values)])
    render('comparison', series, combined=True)
    (OUT / 'data.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
