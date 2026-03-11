import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from skimage.draw import ellipse, polygon, disk
from skimage.filters import gaussian
from scipy.ndimage import binary_dilation, binary_erosion


# HEAD PHANTOM FAMILY

# Head phantom 1: Shepp-Logan phantom
def make_shepp_logan(N=256):
    img = shepp_logan_phantom()
    img = resize(img, (N, N), anti_aliasing=True)
    img = (255 * img / img.max()).astype(np.uint8)
    return img


# Head phantom 2: Custom head phantom similar to the one in the paper
def make_head_phantom(N=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N), dtype=np.float32)
    cx, cy = N // 2, N // 2

    # Head/brain base
    rr, cc = ellipse(cx, cy + 6, int(0.44 * N), int(0.38 * N), shape=img.shape)
    img[rr, cc] = 55

    # Slight asymmetry
    rr, cc = ellipse(cx + 6, cy - 25, int(0.28 * N), int(0.18 * N), shape=img.shape)
    img[rr, cc] = np.maximum(img[rr, cc], 45)

    # Gentle gradient
    yy, xx = np.mgrid[0:N, 0:N]
    grad = 1.0 - 0.7 * np.sqrt(((yy - cx) / (0.55 * N)) ** 2 + ((xx - cy) / (0.55 * N)) ** 2)
    grad = np.clip(grad, 0, 1)
    img *= (0.7 + 0.6 * grad)

    # Skull ring
    rr1, cc1 = ellipse(cx, cy + 6, int(0.46 * N), int(0.40 * N), shape=img.shape)
    rr2, cc2 = ellipse(cx, cy + 6, int(0.43 * N), int(0.37 * N), shape=img.shape)

    skull = np.zeros_like(img, dtype=bool)
    inner = np.zeros_like(img, dtype=bool)
    skull[rr1, cc1] = True
    inner[rr2, cc2] = True
    ring = skull & (~inner)
    img[ring] = 180
    img[inner] = np.maximum(img[inner], 60)

    # Top oval "0"
    rr, cc = ellipse(cx - 55, cy, 26, 16, shape=img.shape)
    img[rr, cc] = 245
    rr, cc = ellipse(cx - 55, cy, 18, 10, shape=img.shape)
    img[rr, cc] = np.minimum(img[rr, cc], 80)

    # Left polygon
    p1_r = np.array([cx + 10, cx + 55, cx + 70, cx + 25])
    p1_c = np.array([cy - 55, cy - 60, cy - 20, cy - 15])
    rr, cc = polygon(p1_r, p1_c, shape=img.shape)
    img[rr, cc] = 245

    # Right polygon
    p2_r = np.array([cx + 30, cx + 55, cx + 75, cx + 58, cx + 72, cx + 40])
    p2_c = np.array([cy + 5,  cy + 40, cy + 60, cy + 55, cy + 85, cy + 60])
    rr, cc = polygon(p2_r, p2_c, shape=img.shape)
    img[rr, cc] = 245

    # CT-like softness
    img = gaussian(img, sigma=1.2, preserve_range=True)
    img += rng.normal(0, 1.2, size=img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


# Head phantom 3: Custom dented gray head phantom
def make_dented_head_phantom(N=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((N, N), dtype=np.float32)
    cx, cy = N // 2, N // 2

    # Main gray head blob
    rr, cc = ellipse(cx, cy, int(0.44 * N), int(0.38 * N), shape=img.shape)
    img[rr, cc] = 70

    # Slight asymmetry / bulges
    rr, cc = ellipse(cx - 8, cy - 18, int(0.18 * N), int(0.10 * N), shape=img.shape)
    img[rr, cc] = np.maximum(img[rr, cc], 78)

    rr, cc = ellipse(cx + 12, cy + 20, int(0.14 * N), int(0.08 * N), shape=img.shape)
    img[rr, cc] = np.maximum(img[rr, cc], 74)

    # Dents
    rr, cc = ellipse(cx - 55, cy + 5, 20, 10, shape=img.shape)
    img[rr, cc] = np.minimum(img[rr, cc], 40)

    rr, cc = ellipse(cx - 20, cy - 58, 10, 8, shape=img.shape)
    img[rr, cc] = np.minimum(img[rr, cc], 42)

    rr, cc = ellipse(cx + 8, cy + 55, 12, 8, shape=img.shape)
    img[rr, cc] = np.minimum(img[rr, cc], 45)

    # Darker inner region
    rr, cc = ellipse(cx, cy, int(0.39 * N), int(0.33 * N), shape=img.shape)
    img[rr, cc] = np.maximum(img[rr, cc], 55)

    # Bright rim
    rr1, cc1 = ellipse(cx, cy, int(0.46 * N), int(0.40 * N), shape=img.shape)
    rr2, cc2 = ellipse(cx, cy, int(0.42 * N), int(0.36 * N), shape=img.shape)

    outer = np.zeros_like(img, dtype=bool)
    inner = np.zeros_like(img, dtype=bool)
    outer[rr1, cc1] = True
    inner[rr2, cc2] = True
    ring = outer & (~inner)
    img[ring] = 170

    # Top ring shape
    rr, cc = ellipse(cx - 50, cy, 20, 12, shape=img.shape)
    img[rr, cc] = 245
    rr, cc = ellipse(cx - 50, cy, 12, 6, shape=img.shape)
    img[rr, cc] = 70

    # Left white shape
    p1_r = np.array([cx + 8, cx + 42, cx + 72, cx + 34])
    p1_c = np.array([cy - 45, cy - 42, cy - 10, cy - 6])
    rr, cc = polygon(p1_r, p1_c, shape=img.shape)
    img[rr, cc] = 245

    # Right white shape
    p2_r = np.array([cx + 18, cx + 38, cx + 68, cx + 50, cx + 66, cx + 34])
    p2_c = np.array([cy + 8, cy + 36, cy + 56, cy + 52, cy + 78, cy + 60])
    rr, cc = polygon(p2_r, p2_c, shape=img.shape)
    img[rr, cc] = 245

    # Gentle shading
    yy, xx = np.mgrid[0:N, 0:N]
    grad = 1.0 - 0.7 * np.sqrt(((yy - cx) / (0.55 * N)) ** 2 + ((xx - cy) / (0.50 * N)) ** 2)
    grad = np.clip(grad, 0, 1)
    img *= (0.75 + 0.45 * grad)

    # Smooth + tiny noise
    img = gaussian(img, sigma=1.2, preserve_range=True)
    img += rng.normal(0, 1.0, size=img.shape)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# RAT FEMUR FAMILY

def make_rat_femur_v1(N=256):
    cx, cy = N // 2, N // 2

    outer = np.zeros((N, N), dtype=bool)
    inner = np.zeros((N, N), dtype=bool)
    trab = np.zeros((N, N), dtype=bool)

    rr, cc = ellipse(cx + 6, cy + 2, 86, 66, rotation=np.deg2rad(-8), shape=outer.shape)
    outer[rr, cc] = True

    rr, cc = ellipse(cx - 30, cy - 10, 30, 18, rotation=np.deg2rad(10), shape=outer.shape)
    outer[rr, cc] = True

    rr, cc = ellipse(cx + 34, cy + 10, 52, 34, rotation=np.deg2rad(6), shape=outer.shape)
    outer[rr, cc] = True

    rr, cc = ellipse(cx + 56, cy + 8, 30, 20, rotation=np.deg2rad(-10), shape=outer.shape)
    outer[rr, cc] = True

    rr, cc = ellipse(cx + 42, cy + 38, 22, 18, rotation=np.deg2rad(15), shape=outer.shape)
    outer[rr, cc] = True

    rr, cc = ellipse(cx + 16, cy + 52, 26, 22, rotation=np.deg2rad(-10), shape=outer.shape)
    outer[rr, cc] = True

    cut1 = np.zeros_like(outer)
    rr, cc = ellipse(cx + 5, cy - 62, 58, 24, rotation=0, shape=outer.shape)
    cut1[rr, cc] = True
    outer = outer & (~cut1)

    for r, c, rad in [
        (cx - 52, cy - 52, 8),
        (cx - 18, cy - 68, 6),
        (cx + 24, cy - 72, 7),
        (cx + 56, cy - 60, 6),
        (cx + 78, cy + 12, 7),
        (cx + 72, cy + 44, 8),
        (cx + 42, cy + 70, 7),
        (cx - 6,  cy + 78, 8),
    ]:
        rr, cc = disk((r, c), rad, shape=outer.shape)
        outer[rr, cc] = False

    pts_r = np.array([cx - 42, cx - 18, cx + 18, cx + 48, cx + 36, cx + 2, cx - 28])
    pts_c = np.array([cy + 6,  cy + 34, cy + 36, cy + 10, cy - 18, cy - 30, cy - 18])
    rr, cc = polygon(pts_r, pts_c, shape=inner.shape)
    inner[rr, cc] = True

    rr, cc = ellipse(cx - 2, cy + 8, 46, 30, rotation=np.deg2rad(-18), shape=inner.shape)
    inner[rr, cc] = True

    rr, cc = ellipse(cx - 18, cy + 2, 22, 18, rotation=np.deg2rad(18), shape=inner.shape)
    inner[rr, cc] = True

    trim = np.zeros_like(inner)
    rr, cc = ellipse(cx + 38, cy - 2, 26, 14, rotation=np.deg2rad(-18), shape=inner.shape)
    trim[rr, cc] = True
    inner = inner & (~trim)

    inner = inner & outer

    ring_outer = np.zeros((N, N), dtype=bool)
    rr, cc = ellipse(cx - 1, cy + 7, 54, 37, rotation=np.deg2rad(-18), shape=ring_outer.shape)
    ring_outer[rr, cc] = True
    cortical_ring = (ring_outer & (~inner)) & outer

    branches = [
        (cx - 25, cy - 2, 18, 3,  35),
        (cx - 10, cy + 8, 20, 3, -25),
        (cx + 4,  cy + 14, 18, 3,  18),
        (cx + 18, cy + 2,  16, 3, -35),
        (cx - 6,  cy - 10, 14, 3,  55),
        (cx - 24, cy + 18, 14, 3, -10),
        (cx + 10, cy - 16, 15, 3,  70),
        (cx + 18, cy + 20, 12, 3, -55),
        (cx - 18, cy - 20, 11, 3,  10),
        (cx - 30, cy + 6,  10, 3,  70),
        (cx - 2,  cy + 26, 12, 3, -5),
        (cx + 8,  cy - 2,  10, 2,  28),
        (cx - 14, cy + 2,   9, 2, -48),
        (cx + 24, cy + 12,  9, 2,  32),
    ]

    for r, c, a, b, ang in branches:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=trab.shape)
        trab[rr, cc] = True

    pores = [
        (cx - 28, cy - 10, 4),
        (cx - 18, cy + 24, 4),
        (cx - 6,  cy - 18, 3),
        (cx + 8,  cy + 30, 3),
        (cx + 24, cy - 8,  4),
        (cx + 28, cy + 18, 3),
        (cx - 30, cy + 16, 3),
        (cx + 4,  cy + 6,  3),
    ]

    for r, c, rad in pores:
        rr, cc = disk((r, c), rad, shape=trab.shape)
        trab[rr, cc] = True

    trab = trab & inner

    rat_femur_three_level = np.zeros((N, N), dtype=np.uint8)
    rat_femur_three_level[outer] = 95
    rat_femur_three_level[cortical_ring] = 160
    rat_femur_three_level[inner] = 235
    rat_femur_three_level[trab] = 150

    return rat_femur_three_level


def make_rat_femur_v2(N=256):
    cx, cy = N // 2, N // 2

    outer = np.zeros((N, N), dtype=bool)
    inner = np.zeros((N, N), dtype=bool)
    trab_dark = np.zeros((N, N), dtype=bool)

    parts = [
        (cx+8,  cy+4,  84, 64, -8),
        (cx-26, cy-6,  30, 18, 10),
        (cx+34, cy+12, 50, 34, 6),
        (cx+56, cy+12, 28, 20, -10),
        (cx+42, cy+42, 24, 18, 10),
        (cx+18, cy+58, 24, 20, -12),
        (cx-8,  cy+66, 16, 20, 0),
        (cx-30, cy+40, 18, 20, -5),
    ]

    for r, c, a, b, ang in parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = True

    cuts = [
        (cx+8,  cy-66, 58, 24, 0),
        (cx-40, cy-54, 12, 10, 0),
        (cx-12, cy-74, 10, 8, 0),
        (cx+20, cy-78, 10, 8, 0),
        (cx+74, cy+14, 10, 10, 0),
        (cx+72, cy+46, 10, 10, 0),
        (cx+44, cy+78, 10, 10, 0),
        (cx-6,  cy+84, 12, 10, 0),
    ]

    for r, c, a, b, ang in cuts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = False

    bumps = [
        (cx-36, cy-20, 12, 10, 20),
        (cx-22, cy+62, 14, 10, -10),
        (cx+18, cy+78, 12, 9, 0),
        (cx+60, cy-10, 12, 10, -20),
    ]

    for r, c, a, b, ang in bumps:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = True

    pts_r = np.array([cx-42, cx-22, cx+10, cx+40, cx+48, cx+28, cx-2, cx-24, cx-38])
    pts_c = np.array([cy+4,  cy+30, cy+40, cy+26, cy+2,  cy-18, cy-34, cy-24, cy-4])
    rr, cc = polygon(pts_r, pts_c, shape=inner.shape)
    inner[rr, cc] = True

    inner_parts = [
        (cx-2,  cy+8,  46, 30, -18),
        (cx-18, cy+2,  22, 18,  20),
        (cx+18, cy+8,  20, 12, -25),
        (cx+6,  cy-14, 16, 10, -15),
    ]

    for r, c, a, b, ang in inner_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=inner.shape)
        inner[rr, cc] = True

    trim_shapes = [
        (cx+42, cy-4, 28, 14, -18),
        (cx-34, cy+24, 10, 8, 0),
    ]

    for r, c, a, b, ang in trim_shapes:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=inner.shape)
        inner[rr, cc] = False

    inner = inner & outer

    ring_outer = np.zeros((N, N), dtype=bool)
    ring_parts = [
        (cx-1, cy+8, 54, 37, -18),
        (cx+10, cy+2, 26, 15, -20),
    ]

    for r, c, a, b, ang in ring_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=ring_outer.shape)
        ring_outer[rr, cc] = True

    cortical_ring = (ring_outer & (~inner)) & outer

    branches = [
        (cx-28, cy-6, 18, 2,  35),
        (cx-18, cy+6, 20, 2, -30),
        (cx-10, cy+18, 12, 2,  20),
        (cx+2,  cy+12, 18, 2, -8),
        (cx+10, cy+2,  18, 2,  32),
        (cx+14, cy-10, 16, 2,  70),
        (cx+20, cy+18, 12, 2, -55),
        (cx-4,  cy-10, 14, 2,  58),
        (cx-24, cy+20, 12, 2, -8),
        (cx-30, cy+6,  10, 2,  72),
        (cx+4,  cy+28, 12, 2, -5),
        (cx+26, cy+10, 10, 2,  28),
        (cx-10, cy-22, 10, 2,  12),
        (cx-2,  cy+2,  10, 2, -45),
        (cx+6,  cy+22, 10, 2,  48),
        (cx-18, cy-8,   9, 2, -65),
        (cx+24, cy-2,   9, 2, -28),
        (cx-30, cy-16,  8, 2,  15),
        (cx-16, cy+30,  9, 2,  60),
    ]

    for r, c, a, b, ang in branches:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=trab_dark.shape)
        trab_dark[rr, cc] = True

    pores = [
        (cx-30, cy-10, 4),
        (cx-22, cy+24, 4),
        (cx-8,  cy-18, 3),
        (cx+6,  cy+30, 3),
        (cx+24, cy-8,  4),
        (cx+28, cy+18, 3),
        (cx-30, cy+14, 3),
        (cx+2,  cy+6,  3),
        (cx-8,  cy+10, 3),
        (cx+12, cy+14, 3),
    ]

    for r, c, rad in pores:
        rr, cc = disk((r, c), rad, shape=trab_dark.shape)
        trab_dark[rr, cc] = True

    trab_dark = trab_dark & inner

    rat_femur_three_level2 = np.zeros((N, N), dtype=np.uint8)
    rat_femur_three_level2[outer] = 92
    rat_femur_three_level2[cortical_ring] = 150
    rat_femur_three_level2[inner] = 238
    rat_femur_three_level2[trab_dark] = 148

    return rat_femur_three_level2


def make_rat_femur_v3(N=256):
    cx, cy = N // 2, N // 2

    outer = np.zeros((N, N), dtype=bool)
    inner = np.zeros((N, N), dtype=bool)
    trab_dark = np.zeros((N, N), dtype=bool)
    trab_bright = np.zeros((N, N), dtype=bool)

    outer_parts = [
        (cx + 6,  cy + 8,  86, 64, -10),
        (cx - 30, cy - 2,  28, 18,  10),
        (cx + 34, cy + 16, 50, 34,   8),
        (cx + 56, cy + 18, 28, 20,  -8),
        (cx + 44, cy + 48, 22, 18,   8),
        (cx + 16, cy + 62, 24, 20, -10),
        (cx - 12, cy + 70, 16, 20,   0),
        (cx - 26, cy + 36, 18, 16,   0),
    ]

    for r, c, a, b, ang in outer_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = True

    left_cuts = [
        (cx + 5,  cy - 67, 60, 23, 0),
        (cx - 52, cy - 44, 16, 10, 0),
        (cx - 40, cy - 5,  20,  8, 0),
        (cx - 18, cy - 72, 10,  8, 0),
        (cx + 18, cy - 78, 10,  8, 0),
        (cx + 74, cy + 16, 10, 10, 0),
        (cx + 70, cy + 48, 10, 10, 0),
        (cx + 42, cy + 78, 10, 10, 0),
        (cx - 4,  cy + 84, 12, 10, 0),
    ]

    for r, c, a, b, ang in left_cuts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = False

    edge_bumps = [
        (cx - 34, cy - 24, 10, 8, 18),
        (cx - 26, cy + 58, 12, 8, -8),
        (cx + 16, cy + 80, 11, 8, 0),
        (cx + 60, cy - 6,  10, 8, -20),
    ]

    for r, c, a, b, ang in edge_bumps:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=outer.shape)
        outer[rr, cc] = True

    pts_r = np.array([cx - 44, cx - 24, cx + 2, cx + 34, cx + 50, cx + 44, cx + 18, cx - 6, cx - 26, cx - 40])
    pts_c = np.array([cy + 6,  cy + 30, cy + 42, cy + 34, cy + 10, cy - 10, cy - 28, cy - 36, cy - 22, cy - 2])
    rr, cc = polygon(pts_r, pts_c, shape=inner.shape)
    inner[rr, cc] = True

    inner_parts = [
        (cx - 2,  cy + 10, 48, 30, -18),
        (cx - 18, cy + 4,  22, 18,  18),
        (cx + 18, cy + 10, 22, 13, -24),
        (cx + 8,  cy - 14, 16, 10, -18),
        (cx - 22, cy - 6,  14, 10,  24),
    ]

    for r, c, a, b, ang in inner_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=inner.shape)
        inner[rr, cc] = True

    trim_parts = [
        (cx + 44, cy - 2, 30, 14, -18),
        (cx - 34, cy + 24, 10,  8,  0),
        (cx + 36, cy + 34, 12,  8,  0),
    ]

    for r, c, a, b, ang in trim_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=inner.shape)
        inner[rr, cc] = False

    inner = inner & outer

    ring_outer = np.zeros((N, N), dtype=bool)
    ring_parts = [
        (cx - 1, cy + 10, 55, 37, -18),
        (cx + 10, cy + 4,  26, 15, -22),
    ]

    for r, c, a, b, ang in ring_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=ring_outer.shape)
        ring_outer[rr, cc] = True

    cortical_ring = (ring_outer & (~inner)) & outer

    dark_branches = [
        (cx - 28, cy - 6,  18, 2,  35),
        (cx - 18, cy + 8,  20, 2, -30),
        (cx - 10, cy + 18, 12, 2,  22),
        (cx + 2,  cy + 12, 18, 2,  -8),
        (cx + 10, cy + 2,  18, 2,  30),
        (cx + 14, cy - 10, 16, 2,  72),
        (cx + 20, cy + 18, 12, 2, -56),
        (cx - 4,  cy - 10, 14, 2,  58),
        (cx - 24, cy + 20, 12, 2, -10),
        (cx - 30, cy + 6,  10, 2,  72),
        (cx + 4,  cy + 28, 12, 2,  -5),
        (cx + 26, cy + 10, 10, 2,  28),
        (cx - 10, cy - 22, 10, 2,  12),
        (cx - 2,  cy + 2,  10, 2, -44),
        (cx + 6,  cy + 22, 10, 2,  48),
        (cx - 18, cy - 8,   9, 2, -65),
        (cx + 24, cy - 2,   9, 2, -28),
        (cx - 30, cy - 16,  8, 2,  15),
        (cx - 16, cy + 30,  9, 2,  60),
        (cx - 4,  cy + 36,  8, 2, -20),
        (cx + 18, cy + 28,  8, 2,  12),
        (cx + 28, cy + 18,  8, 2, -38),
    ]

    for r, c, a, b, ang in dark_branches:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=trab_dark.shape)
        trab_dark[rr, cc] = True

    dark_pores = [
        (cx - 30, cy - 10, 4),
        (cx - 22, cy + 24, 4),
        (cx - 8,  cy - 18, 3),
        (cx + 6,  cy + 30, 3),
        (cx + 24, cy - 8,  4),
        (cx + 28, cy + 18, 3),
        (cx - 30, cy + 14, 3),
        (cx + 2,  cy + 6,  3),
        (cx - 8,  cy + 10, 3),
        (cx + 12, cy + 14, 3),
        (cx - 18, cy + 2,  3),
        (cx + 14, cy - 18, 3),
    ]

    for r, c, rad in dark_pores:
        rr, cc = disk((r, c), rad, shape=trab_dark.shape)
        trab_dark[rr, cc] = True

    trab_dark &= inner

    bright_branches = [
        (cx - 22, cy - 2,  20, 2, -18),
        (cx - 12, cy + 12, 18, 2,  42),
        (cx + 2,  cy + 8,  16, 2, -36),
        (cx + 12, cy - 2,  16, 2,  16),
        (cx + 6,  cy + 24, 14, 2,  30),
        (cx - 8,  cy - 14, 14, 2,  60),
        (cx - 24, cy + 18, 12, 2,  18),
        (cx + 20, cy + 16, 12, 2, -18),
        (cx + 22, cy - 10, 10, 2,  72),
        (cx - 18, cy + 28, 10, 2, -50),
        (cx - 2,  cy + 34, 10, 2,  10),
        (cx + 26, cy + 4,   8, 2,  42),
        (cx - 28, cy - 18,  8, 2, -10),
    ]

    for r, c, a, b, ang in bright_branches:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=trab_bright.shape)
        trab_bright[rr, cc] = True

    bright_nodes = [
        (cx - 24, cy - 8, 3),
        (cx - 10, cy + 20, 3),
        (cx + 6,  cy + 14, 3),
        (cx + 18, cy + 4,  3),
        (cx + 10, cy - 16, 3),
        (cx - 6,  cy - 6,  3),
    ]

    for r, c, rad in bright_nodes:
        rr, cc = disk((r, c), rad, shape=trab_bright.shape)
        trab_bright[rr, cc] = True

    trab_bright &= inner
    trab_bright &= (~trab_dark)

    rat_femur_three_level3 = np.zeros((N, N), dtype=np.uint8)
    rat_femur_three_level3[outer] = 95
    rat_femur_three_level3[cortical_ring] = 155
    rat_femur_three_level3[inner] = 235
    rat_femur_three_level3[trab_dark] = 145
    rat_femur_three_level3[trab_bright] = 248

    return rat_femur_three_level3



# METAL PARTICLES FAMILY

def make_metal_particles_v1(N=256):
    cx, cy = N // 2, N // 2
    rng = np.random.default_rng(7)

    sample = np.zeros((N, N), dtype=bool)
    pores = np.zeros((N, N), dtype=bool)
    particles = np.zeros((N, N), dtype=bool)

    outer_parts = [
        (cx + 8,  cy + 6,  86, 68, -8),
        (cx - 22, cy - 6,  28, 18, 10),
        (cx + 34, cy + 16, 46, 34,  6),
        (cx + 54, cy + 16, 24, 18, -10),
        (cx + 28, cy + 54, 24, 18,  0),
        (cx - 8,  cy + 64, 18, 18,  0),
    ]

    for r, c, a, b, ang in outer_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = True

    cuts = [
        (cx + 4,  cy - 64, 58, 22, 0),
        (cx - 24, cy - 70, 10, 8, 0),
        (cx + 52, cy + 70, 10, 8, 0),
    ]

    for r, c, a, b, ang in cuts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = False

    pore_specs = [
        (cx - 42, cy - 38, 26, 18, -20),
        (cx - 18, cy - 8,  20, 14,  18),
        (cx + 8,  cy - 16, 34, 18,   8),
        (cx + 34, cy - 28, 18, 12, -10),
        (cx - 36, cy + 20, 22, 14,   0),
        (cx - 4,  cy + 18, 18, 12, -18),
        (cx + 26, cy + 18, 16, 12,  20),
        (cx + 46, cy + 30, 14, 10, -25),
        (cx - 22, cy + 52, 18, 12,  10),
        (cx + 8,  cy + 52, 16, 11, -12),
        (cx + 34, cy + 56, 14, 10,  18),
    ]

    for r, c, a, b, ang in pore_specs:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
        pores[rr, cc] = True

    for _ in range(28):
        r = rng.integers(cx - 70, cx + 70)
        c = rng.integers(cy - 45, cy + 55)
        a = rng.integers(4, 10)
        b = rng.integers(3, 8)
        ang = rng.uniform(-40, 40)
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
        pores[rr, cc] = True

    pores &= sample

    pore_dil = binary_dilation(pores, iterations=2)
    pore_er = binary_erosion(pores, iterations=1)
    walls = (pore_dil ^ pore_er) & sample

    wall_glow = binary_dilation(walls, iterations=1) & sample

    wall_coords = np.argwhere(walls)
    choose_n = min(45, len(wall_coords))

    if choose_n > 0:
        idx = rng.choice(len(wall_coords), size=choose_n, replace=False)
        for i in idx:
            r, c = wall_coords[i]
            rad = rng.integers(1, 3)
            rr, cc = disk((r, c), rad, shape=particles.shape)
            particles[rr, cc] = True

    metal_three_level = np.zeros((N, N), dtype=np.uint8)
    metal_three_level[sample] = 35
    metal_three_level[wall_glow] = 90
    metal_three_level[walls] = 165
    metal_three_level[pores] = 8
    metal_three_level[particles] = 255

    return metal_three_level


def make_metal_particles_v2(N=256):
    cx, cy = N // 2, N // 2
    rng = np.random.default_rng(12)

    sample = np.zeros((N, N), dtype=bool)
    pores = np.zeros((N, N), dtype=bool)
    particles = np.zeros((N, N), dtype=bool)

    outer_parts = [
        (cx + 6,  cy + 8,  86, 68, -8),
        (cx - 18, cy - 10, 26, 18, 10),
        (cx + 34, cy + 16, 50, 34,  6),
        (cx + 56, cy + 16, 24, 18, -10),
        (cx + 28, cy + 56, 24, 18,  0),
        (cx - 4,  cy + 64, 18, 18,  0),
    ]

    for r, c, a, b, ang in outer_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = True

    cuts = [
        (cx + 5,  cy - 64, 58, 22, 0),
        (cx - 30, cy - 58, 12, 10, 0),
        (cx + 52, cy + 72, 10, 8, 0),
        (cx - 6,  cy + 82, 12, 10, 0),
    ]

    for r, c, a, b, ang in cuts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = False

    main_pores = [
        (cx - 44, cy - 38, 24, 14, -12),
        (cx - 18, cy - 18, 34, 18,   5),
        (cx + 14, cy - 10, 42, 18,   2),
        (cx + 42, cy - 24, 20, 12, -10),
        (cx - 36, cy + 20, 26, 14,   6),
        (cx - 8,  cy + 18, 22, 12,  -8),
        (cx + 24, cy + 18, 20, 11,  10),
        (cx + 46, cy + 28, 18, 10, -14),
        (cx - 18, cy + 54, 18, 10,   5),
        (cx + 12, cy + 54, 16, 10,  -6),
        (cx + 36, cy + 58, 15,  9,  12),
    ]

    for r, c, a, b, ang in main_pores:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
        pores[rr, cc] = True

    connector_cols = [-28, -8, 14, 34]
    for col in connector_cols:
        for row in [-44, -20, 4, 28, 52]:
            r = cx + row + rng.integers(-4, 5)
            c = cy + col + rng.integers(-4, 5)
            a = rng.integers(6, 12)
            b = rng.integers(4, 8)
            ang = rng.uniform(-20, 20)
            rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
            pores[rr, cc] = True

    for _ in range(26):
        r = rng.integers(cx - 74, cx + 74)
        c = rng.integers(cy - 46, cy + 56)
        rad = rng.integers(3, 7)
        rr, cc = disk((r, c), rad, shape=pores.shape)
        pores[rr, cc] = True

    pores &= sample

    pore_d1 = binary_dilation(pores, iterations=1)
    pore_d2 = binary_dilation(pores, iterations=2)
    pore_e1 = binary_erosion(pores, iterations=1)

    thin_walls = (pore_d1 ^ pores) & sample
    bright_walls = (pore_d2 ^ pore_e1) & sample

    thin_walls &= (~pores)
    bright_walls &= (~pores)

    wall_coords = np.argwhere(bright_walls)
    choose_n = min(38, len(wall_coords))

    if choose_n > 0:
        idx = rng.choice(len(wall_coords), size=choose_n, replace=False)
        for i in idx:
            r, c = wall_coords[i]
            rad = rng.integers(1, 3)
            rr, cc = disk((r, c), rad, shape=particles.shape)
            particles[rr, cc] = True

    metal_soft2 = np.zeros((N, N), dtype=np.float32)
    metal_soft2[sample] = 14
    metal_soft2[thin_walls] = 55
    metal_soft2[bright_walls] = 160
    metal_soft2[pores] = 2
    metal_soft2[particles] = 255

    yy, xx = np.mgrid[0:N, 0:N]

    radial = np.sqrt((yy - (cx - 6))**2 + (xx - (cy + 6))**2)
    radial = radial / radial.max()
    shade = 1.0 - 0.22 * radial

    vertical = 1.03 - 0.10 * (yy / N)
    metal_soft2 *= shade * vertical

    texture = rng.normal(0, 3.0, size=(N, N))
    metal_soft2[sample] += texture[sample]

    metal_soft2 = gaussian(metal_soft2, sigma=0.9, preserve_range=True)

    metal_soft2[bright_walls] += 16
    metal_soft2[thin_walls] += 6
    metal_soft2[pores] -= 4
    metal_soft2[particles] = 255

    metal_soft2 = np.clip(metal_soft2, 0, 255).astype(np.uint8)

    return metal_soft2


def make_metal_particles_v3(N=256):
    rng = np.random.default_rng(21)

    sample = np.zeros((N, N), dtype=bool)
    pores = np.zeros((N, N), dtype=bool)
    particles = np.zeros((N, N), dtype=bool)

    pts_r = np.array([36, 58, 92, 132, 176, 216, 232, 220, 182, 132, 92, 58])
    pts_c = np.array([72, 44, 34, 36, 48, 78, 126, 176, 210, 222, 212, 170])
    rr, cc = polygon(pts_r, pts_c, shape=sample.shape)
    sample[rr, cc] = True

    outer_parts = [
        (118, 120, 84, 58, -12),
        (88,  88,  30, 20,  10),
        (150, 164, 62, 42,   8),
        (196, 132, 30, 42, -12),
    ]
    for r, c, a, b, ang in outer_parts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = True

    cuts = [
        (58,  44, 18, 12,  0),
        (86,  34, 18, 10,  0),
        (216, 64, 16, 12,  0),
        (226, 176, 14, 14, 0),
        (62,  182, 18, 12, 0),
    ]
    for r, c, a, b, ang in cuts:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=sample.shape)
        sample[rr, cc] = False

    large_pores = [
        (70,  62, 18, 10, -20),
        (108, 60, 30, 16, -8),
        (162, 58, 24, 14,  8),
        (92,  102, 18, 12, -6),
        (142, 102, 34, 16, -4),
        (92,  142, 26, 14,  8),
        (150, 144, 22, 12, -10),
        (196, 142, 20, 10,  8),
        (82,  184, 22, 10, 18),
        (130, 184, 16, 10, -8),
        (176, 186, 18, 10, 12),
    ]
    for r, c, a, b, ang in large_pores:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
        pores[rr, cc] = True

    medium_pores = [
        (52,  86, 10,  7,  0),
        (62,  120,  8,  6, 10),
        (74,  154,  8,  6, -6),
        (48,  176, 10,  6, -10),
        (112, 82,   8,  6, 10),
        (126, 122,  8,  6,  0),
        (170, 104, 10,  6, -8),
        (182, 126,  7,  5,  0),
        (204, 164,  8,  5, 15),
        (208, 106,  9,  5, -10),
        (214, 136,  8,  5,  0),
        (222, 118,  7,  5,  0),
    ]
    for r, c, a, b, ang in medium_pores:
        rr, cc = ellipse(r, c, a, b, rotation=np.deg2rad(ang), shape=pores.shape)
        pores[rr, cc] = True

    small_pores = [
        (34,  88, 4), (38, 112, 4), (44, 130, 3), (54, 146, 3),
        (82,  86, 4), (96,  78, 3), (108, 126, 4), (122, 158, 3),
        (144, 80, 4), (156, 120, 3), (170, 168, 4), (188, 100, 3),
        (202, 120, 4), (218, 148, 4), (232, 132, 3), (224, 108, 3),
        (196, 192, 4), (214, 184, 3), (236, 164, 3), (234, 146, 3),
    ]
    for r, c, rad in small_pores:
        rr, cc = disk((r, c), rad, shape=pores.shape)
        pores[rr, cc] = True

    pores &= sample

    d1 = binary_dilation(pores, iterations=1)
    d2 = binary_dilation(pores, iterations=2)
    e1 = binary_erosion(pores, iterations=1)

    thin_walls = (d1 ^ pores) & sample
    bright_walls = (d2 ^ e1) & sample

    thin_walls &= (~pores)
    bright_walls &= (~pores)
    bright_walls = bright_walls & binary_dilation(thin_walls, iterations=1)

    junction_like = bright_walls & binary_dilation(thin_walls, iterations=2)
    coords = np.argwhere(junction_like)
    pick = min(26, len(coords))

    if pick > 0:
        idx = rng.choice(len(coords), size=pick, replace=False)
        for i in idx:
            r, c = coords[i]
            rad = rng.integers(1, 3)
            rr, cc = disk((r, c), rad, shape=particles.shape)
            particles[rr, cc] = True

    extra_pts = [(42, 78), (150, 116), (186, 154), (98, 152), (224, 136)]
    for r, c in extra_pts:
        rr, cc = disk((r, c), 1, shape=particles.shape)
        particles[rr, cc] = True

    particles &= sample

    metal_soft3 = np.zeros((256, 256), dtype=np.float32)
    metal_soft3[sample] = 12
    metal_soft3[thin_walls] = 52
    metal_soft3[bright_walls] = 165
    metal_soft3[pores] = 1.5
    metal_soft3[particles] = 255

    yy, xx = np.mgrid[0:256, 0:256]

    radial = np.sqrt((yy - 120)**2 + (xx - 126)**2)
    radial = radial / radial.max()
    shade = 1.0 - 0.20 * radial

    vertical = 1.02 - 0.08 * (yy / 256)
    metal_soft3 *= shade * vertical

    grain = rng.normal(0, 2.8, size=(256, 256))
    metal_soft3[sample] += grain[sample]

    for col in range(0, 256, 14):
        metal_soft3[:, col:col+1] -= rng.uniform(0.5, 1.5)

    metal_soft3 = gaussian(metal_soft3, sigma=0.85, preserve_range=True)

    metal_soft3[bright_walls] += 18
    metal_soft3[thin_walls] += 5
    metal_soft3[pores] -= 3
    metal_soft3[particles] = 255

    metal_soft3 = np.clip(metal_soft3, 0, 255).astype(np.uint8)

    return metal_soft3


# DENTAL FILLINGS FAMILY

def ellipse_mask(X, Y, x_center, y_center, semi_axis_x, semi_axis_y, angle_deg=0):
    angle_rad = np.deg2rad(angle_deg)
    x_shifted = X - x_center
    y_shifted = Y - y_center
    x_rot = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad)
    y_rot = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)
    return (x_rot**2 / semi_axis_x**2 + y_rot**2 / semi_axis_y**2) <= 1.0


def gaussian_blur(img, sigma=1.4, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2

    kernel_x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(kernel_x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    padded_img = np.pad(img, ((radius, radius), (radius, radius)), mode="edge")

    row_blurred = np.zeros_like(padded_img, dtype=np.float32)
    for row_idx in range(padded_img.shape[0]):
        row_blurred[row_idx, :] = np.convolve(padded_img[row_idx, :], kernel, mode="same")

    col_blurred = np.zeros_like(row_blurred, dtype=np.float32)
    for col_idx in range(row_blurred.shape[1]):
        col_blurred[:, col_idx] = np.convolve(row_blurred[:, col_idx], kernel, mode="same")

    return col_blurred[radius:-radius, radius:-radius]


def add_disk(img, X, Y, x_center, y_center, radius, value):
    disk_mask = (X - x_center)**2 + (Y - y_center)**2 <= radius**2
    img[disk_mask] = value
    return disk_mask


def generate_dental_phantom(seed=0, N=512):
    rng = np.random.default_rng(seed)

    phantom = np.zeros((N, N), dtype=np.float32)

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    head_center_x = rng.uniform(-0.03, 0.03)
    head_center_y = rng.uniform(0.02, 0.10)
    head_semi_x   = rng.uniform(0.86, 0.92)
    head_semi_y   = rng.uniform(1.00, 1.08)

    skull_thickness = rng.uniform(0.03, 0.06)

    soft_tissue_val = rng.uniform(0.18, 0.24)
    skin_ring_val   = rng.uniform(0.25, 0.32)
    skull_val       = rng.uniform(0.78, 0.92)
    brain_val       = rng.uniform(0.40, 0.52)

    jaw_shift_x = rng.uniform(-0.05, 0.05)
    jaw_angle_deg = rng.uniform(10, 22)
    jaw_semi_x = rng.uniform(0.10, 0.14)

    blur_sigma = rng.uniform(1.3, 2.1)
    shading_strength = rng.uniform(0.25, 0.38)

    head_outer_mask = ellipse_mask(X, Y, head_center_x, head_center_y, head_semi_x, head_semi_y)
    head_inner_mask = ellipse_mask(X, Y, head_center_x, head_center_y, head_semi_x - 0.04, head_semi_y - 0.04)
    skin_ring_mask  = head_outer_mask & (~head_inner_mask)

    phantom[head_outer_mask] = np.maximum(phantom[head_outer_mask], soft_tissue_val)
    phantom[skin_ring_mask]  = np.maximum(phantom[skin_ring_mask], skin_ring_val)

    skull_outer_mask = ellipse_mask(X, Y, head_center_x, head_center_y, head_semi_x - 0.04, head_semi_y - 0.04)
    skull_inner_mask = ellipse_mask(
        X, Y, head_center_x, head_center_y,
        head_semi_x - 0.04 - skull_thickness,
        head_semi_y - 0.04 - skull_thickness
    )
    skull_ring_mask = skull_outer_mask & (~skull_inner_mask)
    phantom[skull_ring_mask] = skull_val

    brain_mask = ellipse_mask(X, Y, head_center_x, head_center_y + 0.03, head_semi_x - 0.10, head_semi_y - 0.14)
    phantom[brain_mask] = np.maximum(phantom[brain_mask], brain_val)

    texture = rng.normal(0, 0.015, size=phantom.shape).astype(np.float32)
    phantom[head_outer_mask] = np.clip(phantom[head_outer_mask] + texture[head_outer_mask], 0, 1.5)

    jaw_left_mask  = ellipse_mask(X, Y, -0.35 + jaw_shift_x, -0.20, jaw_semi_x, 0.45, angle_deg=jaw_angle_deg)
    jaw_right_mask = ellipse_mask(X, Y,  0.35 + jaw_shift_x, -0.20, jaw_semi_x, 0.45, angle_deg=-jaw_angle_deg)
    phantom[jaw_left_mask]  = np.maximum(phantom[jaw_left_mask],  rng.uniform(0.68, 0.80))
    phantom[jaw_right_mask] = np.maximum(phantom[jaw_right_mask], rng.uniform(0.68, 0.80))

    center_bone_mask = ellipse_mask(
        X, Y,
        rng.uniform(-0.03, 0.03), -0.35,
        rng.uniform(0.15, 0.22), rng.uniform(0.12, 0.18)
    )
    phantom[center_bone_mask] = np.maximum(phantom[center_bone_mask], rng.uniform(0.55, 0.72))

    air_offset_x = rng.uniform(0.06, 0.10)
    air_center_y = rng.uniform(-0.08, -0.02)
    air_left_mask  = ellipse_mask(X, Y, -air_offset_x, air_center_y, 0.05, 0.025)
    air_right_mask = ellipse_mask(X, Y,  air_offset_x, air_center_y, 0.05, 0.025)
    phantom[air_left_mask | air_right_mask] = rng.uniform(0.10, 0.18)

    tooth_arc_center_x, tooth_arc_center_y = 0.00, -0.68
    tooth_arc_radius = 0.22
    num_teeth = 12
    tooth_angles = np.deg2rad(np.linspace(200, 340, num_teeth))

    teeth_mask_total = np.zeros_like(phantom, dtype=bool)
    fillings_mask_total = np.zeros_like(phantom, dtype=bool)

    for theta in tooth_angles:
        tooth_center_x = tooth_arc_center_x + tooth_arc_radius * np.cos(theta)
        tooth_center_y = tooth_arc_center_y + 0.30 * tooth_arc_radius * np.sin(theta)

        enamel_val = rng.uniform(0.75, 1.00)
        dentin_val = rng.uniform(0.55, 0.75)

        enamel_radius = rng.uniform(0.020, 0.028)
        enamel_mask = add_disk(phantom, X, Y, tooth_center_x, tooth_center_y, radius=enamel_radius, value=enamel_val)

        dentin_radius = enamel_radius * rng.uniform(0.55, 0.75)
        dentin_mask = add_disk(phantom, X, Y, tooth_center_x, tooth_center_y, radius=dentin_radius, value=dentin_val)

        if rng.random() < 0.55:
            filling_val = rng.uniform(1.15, 1.50)
            filling_radius = dentin_radius * rng.uniform(0.35, 0.60)
            filling_mask = add_disk(
                phantom, X, Y, tooth_center_x, tooth_center_y,
                radius=filling_radius, value=filling_val
            )
            fillings_mask_total |= filling_mask

        teeth_mask_total |= enamel_mask | dentin_mask

    radial_dist = np.sqrt((X * 0.95)**2 + (Y * 1.05)**2)
    shading = np.clip(1.0 - shading_strength * radial_dist, 0.70, 1.0)
    phantom *= shading

    blurred = gaussian_blur(phantom, sigma=blur_sigma)
    blurred[teeth_mask_total] = np.maximum(blurred[teeth_mask_total], 0.65)
    blurred[fillings_mask_total] = 1.50

    return blurred


# PLOTTING

def show_family(images, titles, family_title, cmap="gray", vmin=None, vmax=None, figsize=(15, 5)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(family_title, fontsize=14)

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Head family
    head_imgs = [
        make_shepp_logan(256),
        make_head_phantom(256, seed=1),
        make_dented_head_phantom(256, seed=2),
    ]
    head_titles = [
        "(Shepp-Logan) head phantom v1",
        "Custom head phantom v2",
        "Custom head phantom v3",
    ]
    show_family(head_imgs, head_titles, "Head phantom family", cmap="gray", vmin=0, vmax=255)

    # Rat femur family
    rat_imgs = [
        make_rat_femur_v1(256),
        make_rat_femur_v2(256),
        make_rat_femur_v3(256),
    ]
    rat_titles = [
        "Rat femur phantom v1",
        "Rat femur phantom v2",
        "Rat femur phantom v3",
    ]
    show_family(rat_imgs, rat_titles, "Rat femur phantom family", cmap="gray", vmin=0, vmax=255)

    # Metal particles family
    metal_imgs = [
        make_metal_particles_v1(256),
        make_metal_particles_v2(256),
        make_metal_particles_v3(256),
    ]
    metal_titles = [
        "Metal particles phantom v1",
        "Metal particles phantomv2",
        "Metal particles phantomv3",
    ]
    show_family(metal_imgs, metal_titles, "Metal particles phantom family", cmap="gray", vmin=0, vmax=255)

    # Dental family - keep only 3 variations
    dental_imgs = [
        generate_dental_phantom(seed=0, N=512),
        generate_dental_phantom(seed=1, N=512),
        generate_dental_phantom(seed=2, N=512),
    ]
    dental_titles = [
        "Dental fillings phantom v1",
        "Dental fillings phantom v2",
        "Dental fillings phantom v3",
    ]
    show_family(dental_imgs, dental_titles, "Dental fillings phantom family", cmap="gray", vmin=0, vmax=1.5, figsize=(15, 5))