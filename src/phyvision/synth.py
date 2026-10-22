"""Procedural (AI-generated) human physique renderer.

Every image in the PhyVision dataset is produced by this module. A person is
modelled as a set of anatomical width/length parameters measured as fractions
of body height. Each physique class draws those parameters from its own
distribution (with deliberate overlap at class borders), then the renderer
adds clothing, skin tone, hair, pose, lighting, background clutter, camera
tilt and sensor noise. Because the parameters are known, every image comes with
an exact ground-truth label and a pixel-perfect segmentation mask.

Version 5.1 adds per-arm poses (bicep flex, raised arm, hands on hips, hand in
pocket), slight 3/4 body turns, photo framings from head-to-feet down to
waist-up, variable portrait aspect ratios, plain stock-photo backdrops and
faint watermark overlays.

AI assistance: this dataset generator was built with help from Claude, an AI
model by Anthropic, as a learning aid for creating computer-vision training data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

# Palettes
SKIN_TONES = [
    (255, 224, 196), (241, 194, 163), (234, 189, 157), (224, 172, 135),
    (198, 134, 96), (178, 120, 82), (160, 102, 70), (141, 85, 56),
    (115, 70, 46), (92, 56, 38), (74, 46, 32), (232, 190, 172),
]
HAIR_COLOURS = [
    (20, 18, 18), (45, 32, 25), (80, 55, 35), (120, 85, 50), (170, 130, 80),
    (215, 185, 120), (150, 60, 35), (140, 140, 140), (230, 230, 225),
]

# Body parameter model
# Base half-widths (fraction of body height) for an average adult.
# Calibrated in v5.1 against adult anthropometric ratios (stature-relative
# breadths from ANSUR II style survey data): e.g. for a 1.75 m man the normal
# upper arm is ~9.8 cm across (~31 cm circumference), mid-thigh ~17.5 cm,
# waist breadth ~30 cm, outer shoulder breadth ~47 cm, head breadth ~16 cm.
BASE = {
    "male":   dict(S=0.128, C=0.104, W=0.085, P=0.096, neck=0.030,
                   a1=0.028, a2=0.023, t1=0.050, t2=0.033, head=1.00),
    "female": dict(S=0.113, C=0.097, W=0.076, P=0.106, neck=0.026,
                   a1=0.025, a2=0.020, t1=0.052, t2=0.032, head=0.96),
}

# Multiplicative class modifiers (mean, sd) applied on top of BASE.
# Realistic, not caricatured: a skinny adult keeps most of the bone frame
# (shoulders ~7% narrower) but has much thinner limbs (~24 cm upper-arm
# circumference) and a narrow waist; athletic adds muscle to shoulders and
# limbs with a normal waist; obese widens mainly the waist, belly and limbs.
CLASS_MODS = {
    "skinny": dict(S=(0.93, .035), C=(0.86, .035), W=(0.84, .035), P=(0.88, .035),
                   neck=(0.88, .05), arms=(0.76, .045), legs=(0.80, .045),
                   belly=(0.000, .002), head=(0.98, .02)),
    "normal": dict(S=(1.00, .035), C=(1.00, .035), W=(1.00, .04), P=(1.00, .035),
                   neck=(1.00, .05), arms=(1.00, .055), legs=(1.00, .05),
                   belly=(0.004, .003), head=(1.00, .02)),
    "athletic": dict(S=(1.14, .04), C=(1.10, .04), W=(0.95, .035), P=(0.98, .035),
                     neck=(1.22, .06), arms=(1.30, .06), legs=(1.18, .055),
                     belly=(0.000, .002), head=(1.00, .02)),
    "obese": dict(S=(1.16, .05), C=(1.36, .06), W=(1.60, .09), P=(1.40, .07),
                  neck=(1.40, .07), arms=(1.38, .08), legs=(1.38, .07),
                  belly=(0.040, .012), head=(1.10, .03)),
}


@dataclass
class BodySpec:
    label: str
    sex: str
    S: float
    C: float
    W: float
    P: float
    neck: float
    a1: float
    a2: float
    t1: float
    t2: float
    belly: float
    head: float
    arm_pose: str
    arm_angle: float
    elbow_bend: float
    leg_spread: float
    view_squash: float
    arms_lr: tuple = (("down", 0.1, 0.0), ("down", 0.1, 0.0))  # per side: (kind, upper angle, forearm angle/bend)
    turn: float = 0.0             # >0 narrows the right half (3/4 view), <0 the left
    skin: tuple = field(default=(200, 150, 120))
    hair: tuple = field(default=(30, 25, 20))
    hair_style: str = "short"
    top: str = "tshirt"           # none | tank | tshirt | long
    bottom: str = "shorts"        # shorts | capri | pants
    loose: float = 0.0            # clothing looseness (fraction of height)
    top_colour: tuple = (40, 90, 160)
    bottom_colour: tuple = (40, 40, 50)
    shoe_colour: tuple = (30, 30, 30)

    def to_dict(self):
        d = asdict(self)
        d["shoulder_waist_ratio"] = round(self.S / self.W, 3)
        d["waist_hip_ratio"] = round(self.W / self.P, 3)
        return d


def _rand_colour(rng, sat=(0.25, 0.85), val=(0.25, 0.95)):
    h = rng.uniform(0, 180)
    s = rng.uniform(*sat) * 255
    v = rng.uniform(*val) * 255
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(c) for c in rgb)


def sample_body(label: str, rng: np.random.Generator, sex: str | None = None, pose: str | None = None) -> BodySpec:
    r = rng.random()
    sex = sex or ("male" if r < 0.55 else "female")
    base = BASE[sex]
    mods = CLASS_MODS[label]
    g = lambda key: max(0.4, rng.normal(*mods[key]))
    size = rng.normal(1.0, 0.025)  # shared frame-size factor

    arms = g("arms")
    legs = g("legs")
    S = base["S"] * g("S") * size
    W = base["W"] * g("W") * size
    if label == "athletic" and sex == "female":
        S *= 0.97  # women athletes: less extreme V-taper
    spec = BodySpec(
        label=label, sex=sex,
        S=S, C=base["C"] * g("C") * size, W=W, P=base["P"] * g("P") * size,
        neck=base["neck"] * g("neck"),
        a1=base["a1"] * arms * rng.normal(1, .04),
        a2=base["a2"] * arms * rng.normal(1, .04) ** 0.8,
        t1=base["t1"] * legs * rng.normal(1, .04),
        t2=base["t2"] * legs * rng.normal(1, .04),
        belly=max(0.0, rng.normal(*mods["belly"])),
        head=base["head"] * g("head"),
        arm_pose=rng.choice(["down", "down", "apart", "apart", "hips"], p=[.3, .15, .25, .15, .15]),
        arm_angle=0.0, elbow_bend=0.0,
        leg_spread=rng.uniform(0.0, 0.07),
        view_squash=rng.uniform(0.93, 1.0),
    )
    def arm(kind):
        if kind == "down":
            a = rng.uniform(0.04, 0.22)
            return ("down", a, a + rng.uniform(-0.05, 0.12))
        if kind == "apart":
            a = rng.uniform(0.22, 0.55)
            return ("apart", a, a + rng.uniform(-0.15, 0.1))
        if kind == "hips":
            return ("hips", rng.uniform(0.55, 0.85), 0.0)
        if kind == "pocket":
            return ("pocket", rng.uniform(0.10, 0.30), 0.0)
        if kind == "flex":  # upper arm out to the side, forearm up (bicep pose)
            return ("flex", rng.uniform(1.25, 1.80), rng.uniform(2.70, 3.35))
        a = rng.uniform(2.55, 3.05)  # raised overhead
        return ("raise", a, a + rng.uniform(-0.15, 0.15))

    mode = rng.choice(["sym", "one_flex", "double_flex", "one_raise", "both_raise", "mixed"],
                      p=[0.52, 0.16, 0.10, 0.08, 0.04, 0.10])
    mode = pose or mode
    if mode in ("down", "apart", "hips", "pocket"):  # forced symmetric pose (used for demo people)
        lr = (arm(mode), arm(mode))
        mode = "sym"
    elif mode == "sym":
        k = rng.choice(["down", "apart", "hips", "pocket"], p=[0.38, 0.34, 0.18, 0.10])
        lr = (arm(k), arm(k))
    elif mode == "one_flex":
        lr = (arm("flex"), arm(rng.choice(["down", "pocket", "hips", "apart"])))
    elif mode == "double_flex":
        lr = (arm("flex"), arm("flex"))
    elif mode == "one_raise":
        lr = (arm("raise"), arm(rng.choice(["down", "apart", "hips"])))
    elif mode == "both_raise":
        lr = (arm("raise"), arm("raise"))
    else:
        ks = rng.choice(["down", "apart", "hips", "pocket"], 2, replace=False)
        lr = (arm(ks[0]), arm(ks[1]))
    if rng.random() < 0.5:
        lr = (lr[1], lr[0])
    spec.arms_lr = lr
    spec.arm_pose = mode if mode != "sym" else lr[0][0]
    spec.arm_angle, spec.elbow_bend = float(lr[0][1]), float(lr[0][2])
    spec.turn = float(rng.uniform(-0.18, 0.18)) if rng.random() < 0.4 else 0.0

    # Appearance
    spec.skin = SKIN_TONES[rng.integers(len(SKIN_TONES))]
    spec.hair = HAIR_COLOURS[rng.integers(len(HAIR_COLOURS))]
    spec.hair_style = rng.choice(["short", "short", "long", "bun", "bald"]) if sex == "male" \
        else rng.choice(["long", "long", "bun", "short"])
    if sex == "male" and spec.hair_style == "long":
        spec.hair_style = "short"
    shirtless_p = {"athletic": 0.35, "skinny": 0.12, "normal": 0.10, "obese": 0.10}[label]
    if sex == "female":
        spec.top = rng.choice(["tank", "tshirt", "tshirt", "long", "sports"])
    elif rng.random() < shirtless_p:
        spec.top = "none"
    else:
        spec.top = rng.choice(["tank", "tshirt", "tshirt", "long"])
    spec.bottom = rng.choice(["shorts", "capri", "pants", "pants"])
    if spec.top == "none":
        spec.bottom = rng.choice(["shorts", "shorts", "pants"])
    # Loose clothing hides shape a little (realistic confound)
    spec.loose = abs(rng.normal(0, 0.006)) if rng.random() < 0.55 else 0.0
    spec.top_colour = _rand_colour(rng)
    spec.bottom_colour = _rand_colour(rng, sat=(0.05, 0.6), val=(0.12, 0.7))
    spec.shoe_colour = _rand_colour(rng, sat=(0.0, 0.5), val=(0.08, 0.9))
    return spec


# Geometry helpers
def _catmull_rom(points: np.ndarray, samples: int = 10) -> np.ndarray:
    """Smooth an open polyline with a Catmull-Rom spline."""
    pts = np.vstack([points[0], points, points[-1]])
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(points[-1])
    return np.array(out)


def _capsule_chain(mask, pts, radii, upto: float = 1.0):
    """Draw a tapered limb through `pts` with per-point `radii` (pixels).

    `upto` (0..1) limits the drawn fraction of the chain length (used for sleeves
    and trouser legs)."""
    pts = np.asarray(pts, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    limit = total * upto
    run = 0.0
    for i in range(len(seg)):
        n = max(2, int(seg[i] / 1.5))
        for t in np.linspace(0, 1, n):
            if run + t * seg[i] > limit:
                return
            p = pts[i] + (pts[i + 1] - pts[i]) * t
            r = radii[i] + (radii[i + 1] - radii[i]) * t
            cv2.circle(mask, (int(round(p[0])), int(round(p[1]))), max(1, int(round(r))), 255, -1, cv2.LINE_AA)
        run += seg[i]


class _Canvas:
    def __init__(self, w, h, cx, top, height):
        self.w, self.h = w, h
        self.cx, self.top, self.H = cx, top, height

    def p(self, xn, yn):
        return np.array([self.cx + xn * self.H, self.top + yn * self.H], dtype=np.float32)

    def r(self, rn):
        return rn * self.H

    def blank(self):
        return np.zeros((self.h, self.w), np.uint8)


def _body_parts(spec: BodySpec, cv: _Canvas):
    """Return dict of part masks (uint8) for the body and clothing regions."""
    sq = spec.view_squash
    S, C, W, P = spec.S * sq, spec.C * sq, spec.W * sq, spec.P * sq
    belly = spec.belly * sq
    neck = spec.neck * sq

    parts = {}
    # ---- torso -------------------------------------------------------------
    armpit = max(C * 0.97, W)
    belly_w = max(W, (W + P) / 2) + belly
    right = np.array([
        [neck * 1.05, 0.150],
        [S * 0.80, 0.168],
        [S, 0.196],
        [armpit, 0.236],
        [C, 0.272],
        [(C + W) / 2 + belly * 0.4, 0.335],
        [W + belly * 0.7, 0.400],
        [belly_w, 0.452],
        [P + belly * 0.3, 0.500],
        [P * 0.93, 0.535],
        [0.0, 0.545],
    ])
    curve = _catmull_rom(right, 8)
    left = curve[::-1].copy()
    left[:, 0] *= -1
    outline = np.vstack([curve, left])
    poly = np.array([cv.p(x, y) for x, y in outline], dtype=np.int32)
    torso = cv.blank()
    cv2.fillPoly(torso, [poly], 255, cv2.LINE_AA)
    parts["torso"] = torso

    # ---- head + neck ---------------------------------------------------------
    head = cv.blank()
    hw = 0.047 * spec.head * (0.96 + 0.04 * sq)
    hc = cv.p(0, 0.066)
    cv2.ellipse(head, (int(hc[0]), int(hc[1])), (int(cv.r(hw)), int(cv.r(0.062))), 0, 0, 360, 255, -1, cv2.LINE_AA)
    if spec.label == "obese":  # double chin / fuller jaw
        jc = cv.p(0, 0.108)
        cv2.ellipse(head, (int(jc[0]), int(jc[1])), (int(cv.r(hw * 0.85)), int(cv.r(0.025))), 0, 0, 360, 255, -1, cv2.LINE_AA)
    n0, n1 = cv.p(-neck, 0.105), cv.p(neck, 0.165)
    cv2.rectangle(head, (int(n0[0]), int(n0[1])), (int(n1[0]), int(n1[1])), 255, -1)
    parts["head"] = head

    # ---- arms ----------------------------------------------------------------
    arms = cv.blank()
    sleeve = cv.blank()
    sleeve_frac = {"none": 0.0, "tank": 0.0, "sports": 0.0, "tshirt": 0.38, "long": 0.93}[spec.top]
    for side, (kind, ang, fa) in zip((-1, 1), spec.arms_lr):
        sh = cv.p(side * (S - spec.a1 * 0.85 * sq), 0.205)
        up_len, fo_len, hand_len = 0.185, 0.158, 0.075
        if kind in ("flex", "raise"):
            sh = sh + np.array([0, -cv.r(0.012)])  # shoulder lifts when the arm goes up
        elbow = sh + cv.r(up_len) * np.array([side * math.sin(ang), math.cos(ang)])
        hand_r = 0.017
        if kind in ("hips", "pocket"):
            # forearm points back toward the hip / front pocket
            tgt = cv.p(side * (P + belly * 0.3) * (0.95 if kind == "hips" else 0.70), 0.47 if kind == "hips" else 0.52)
            d = tgt - elbow
            d = d / (np.linalg.norm(d) + 1e-6)
            wrist = elbow + d * cv.r(fo_len * (0.95 if kind == "hips" else 0.85))
            hand = wrist + d * cv.r(hand_len * (0.5 if kind == "hips" else 0.25))
        else:
            wrist = elbow + cv.r(fo_len) * np.array([side * math.sin(fa), math.cos(fa)])
            hand = wrist + cv.r(hand_len * (0.55 if kind == "flex" else 1.0)) * np.array([side * math.sin(fa) * 0.9, math.cos(fa)])
            if kind == "flex":
                hand_r = 0.022  # clenched fist
        pts = [sh, (sh + elbow) / 2, elbow, (elbow + wrist) / 2, wrist, hand]
        a1, a2 = spec.a1 * sq, spec.a2 * sq
        bulge = 1.0
        if kind == "flex":  # contracted biceps
            bulge = {"athletic": 1.28, "normal": 1.10, "obese": 1.04, "skinny": 1.03}[spec.label]
        rad = [cv.r(a1 * 1.12), cv.r(a1 * bulge), cv.r(a1 * 0.72), cv.r(a2), cv.r(a2 * 0.6), cv.r(hand_r)]
        _capsule_chain(arms, pts, rad)
        if sleeve_frac > 0:
            _capsule_chain(sleeve, pts, [r + cv.r(spec.loose + 0.004) for r in rad], upto=sleeve_frac)
    parts["arms"] = arms

    # ---- legs ----------------------------------------------------------------
    legs = cv.blank()
    trouser = cv.blank()
    shoes = cv.blank()
    leg_frac = {"shorts": 0.33, "capri": 0.68, "pants": 0.97}[spec.bottom]
    t1, t2 = spec.t1 * sq, spec.t2 * sq
    for side in (-1, 1):
        top = cv.p(side * max(P * 0.93 - t1 * 0.95, t1 * 0.55), 0.50)
        knee = cv.p(side * (P * 0.42 + spec.leg_spread * 0.6), 0.735)
        ankle = cv.p(side * (P * 0.36 + spec.leg_spread), 0.955)
        calf = knee + (ankle - knee) * 0.32
        pts = [top, (top + knee) / 2, knee, calf, ankle]
        rad = [cv.r(t1), cv.r(t1 * 0.88), cv.r(t2 * 0.82), cv.r(t2), cv.r(t2 * 0.45)]
        _capsule_chain(legs, pts, rad)
        _capsule_chain(trouser, pts, [r + cv.r(spec.loose * 0.8 + 0.003) for r in rad], upto=leg_frac)
        foot_c = cv.p(side * (P * 0.36 + spec.leg_spread + 0.018), 0.978)
        cv2.ellipse(shoes, (int(foot_c[0]), int(foot_c[1])), (int(cv.r(0.040)), int(cv.r(0.019))),
                    side * 12, 0, 360, 255, -1, cv2.LINE_AA)
    parts["legs"] = legs
    parts["shoes"] = shoes

    body = np.maximum.reduce([parts["torso"], parts["head"], arms, legs, shoes])

    # ---- clothing regions ------------------------------------------------------
    ys = np.arange(cv.h)[:, None]
    def band(y0, y1):
        a, b = cv.top + y0 * cv.H, cv.top + y1 * cv.H
        return ((ys >= a) & (ys <= b)).astype(np.uint8) * 255 * np.ones((1, cv.w), np.uint8)

    loose_px = int(round(cv.r(spec.loose))) + 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * loose_px + 1, 2 * loose_px + 1))
    top_m = np.zeros_like(body)
    if spec.top != "none":
        top_end = {"tank": 0.50, "tshirt": 0.52, "long": 0.53, "sports": 0.33}[spec.top]
        top_m = cv2.bitwise_and(parts["torso"], band(0.15, top_end))
        if spec.top != "sports":
            top_m = cv2.dilate(top_m, kern)
            top_m = cv2.bitwise_and(top_m, band(0.15, top_end + spec.loose))
        top_m = np.maximum(top_m, sleeve)
    bottom_m = cv2.bitwise_and(parts["torso"], band(0.44, 0.56))
    bottom_m = np.maximum(bottom_m, trouser)
    bottom_m = cv2.dilate(bottom_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bottom_m = cv2.bitwise_and(bottom_m, band(0.44, 0.97))

    parts["top"] = top_m
    parts["bottom"] = bottom_m
    parts["silhouette"] = np.maximum.reduce([body, top_m, bottom_m])
    return parts


# Backgrounds
_WM_WORDS = ["stock", "sample", "preview", "photo", "image", "royalty", "demo", "watermark"]


def _watermark(img, rng):
    """Faint stock-photo style text overlay."""
    h, w = img.shape[:2]
    layer = img.copy()
    word = rng.choice(_WM_WORDS) + (rng.choice(["", ".com", " ID 123456"]))
    scale = w / 260 * rng.uniform(0.6, 1.2)
    col = (255, 255, 255) if rng.random() < 0.6 else (150, 150, 150)
    for _ in range(int(rng.integers(1, 4))):
        org = (int(rng.uniform(0, 0.4) * w), int(rng.uniform(0.2, 0.9) * h))
        cv2.putText(layer, word, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, max(1, int(scale * 2)), cv2.LINE_AA)
    a = rng.uniform(0.12, 0.35)
    return cv2.addWeighted(layer, a, img, 1 - a, 0)


def _background(w, h, rng):
    kind = rng.choice(["plain", "studio", "room", "outdoor", "gym"], p=[0.25, 0.2, 0.2, 0.2, 0.15])
    img = np.zeros((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h)[:, None, None]
    if kind == "plain":  # seamless white / light-grey stock-photo backdrop
        v = rng.uniform(225, 255)
        img[:] = v - rng.uniform(0, 18) * yy
    elif kind == "studio":
        c1 = np.array(_rand_colour(rng, (0, .3), (.55, .98)), np.float32)
        c2 = c1 * rng.uniform(0.55, 0.9)
        img[:] = c1 * (1 - yy) + c2 * yy
    elif kind == "outdoor":
        sky = np.array(_rand_colour(rng, (.2, .6), (.7, 1.0)), np.float32)
        ground = np.array(_rand_colour(rng, (.2, .7), (.25, .6)), np.float32)
        horizon = rng.uniform(0.45, 0.7)
        img[:] = np.where(yy < horizon, sky * (0.8 + 0.2 * yy), ground * (0.7 + 0.3 * yy))
        for _ in range(rng.integers(2, 7)):  # trees / bushes
            c = tuple(float(v) for v in _rand_colour(rng, (.3, .8), (.15, .5)))
            cx, cy = int(rng.uniform(0, w)), int(h * horizon)
            cv2.ellipse(img, (cx, cy), (int(rng.uniform(.08, .25) * w), int(rng.uniform(.1, .3) * h)),
                        0, 180, 360, c, -1)
    else:  # room / gym
        wall = np.array(_rand_colour(rng, (0, .35), (.45, .95)), np.float32)
        floor = np.array(_rand_colour(rng, (0, .4), (.2, .6)), np.float32)
        fl = rng.uniform(0.72, 0.86)
        img[:] = np.where(yy < fl, wall, floor)
        for _ in range(rng.integers(1, 5)):  # doors, frames, furniture, racks
            c = tuple(float(v) for v in _rand_colour(rng, (0, .7), (.15, .9)))
            x0, y0 = int(rng.uniform(-.1, .9) * w), int(rng.uniform(.05, .7) * h)
            x1, y1 = x0 + int(rng.uniform(.08, .35) * w), y0 + int(rng.uniform(.08, .5) * h)
            cv2.rectangle(img, (x0, y0), (x1, min(y1, int(fl * h))), c, -1)
        if kind == "gym":
            for _ in range(rng.integers(1, 4)):
                y = int(rng.uniform(.2, .6) * h)
                cv2.line(img, (0, y), (w, y), (60, 60, 60), max(1, h // 120))
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 3.0))  # depth-of-field
    vign = 1 - 0.35 * rng.random() * ((np.linspace(-1, 1, w)[None, :] ** 2 + np.linspace(-1, 1, h)[:, None] ** 2) / 2)
    img *= vign[..., None]
    return img, kind


# Rendering
FRAMINGS = {  # name: (probability, visible fraction of body height range)
    "full": (0.50, (1.0, 1.0)),
    "knees": (0.15, (0.72, 0.82)),
    "thigh": (0.20, (0.57, 0.70)),
    "waist": (0.15, (0.47, 0.56)),
}


def sample_framing(rng):
    names = list(FRAMINGS)
    name = rng.choice(names, p=[FRAMINGS[n][0] for n in names])
    lo, hi = FRAMINGS[name][1]
    return str(name), float(rng.uniform(lo, hi))


def framing_name(visible: float) -> str:
    if visible >= 0.93:
        return "full"
    if visible >= 0.72:
        return "knees"
    if visible >= 0.57:
        return "thigh"
    return "waist"


def _turn_warp(img, sil, cx, turn):
    """Squeeze one half of the body horizontally to mimic a 3/4 turn."""
    h, w = sil.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    f = 1.0 - abs(turn)
    src = xs.copy()
    if turn > 0:  # right half narrower
        m = xs > cx
        src[m] = cx + (xs[m] - cx) / f
    else:
        m = xs < cx
        src[m] = cx - (cx - xs[m]) / f
    mx = np.repeat(src[None, :], h, 0)
    my = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
    return (cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=0),
            cv2.remap(sil, mx, my, cv2.INTER_LINEAR, borderValue=0))


def render(spec: BodySpec, rng: np.random.Generator, out_w=None, out_h=288, ss=3,
           body_frac=None, with_background=True, detail=True, framing=None, watermark=True):
    """Render a person. Returns (rgb uint8 HxWx3, mask uint8 HxW 0/255, meta dict).

    framing: None (random), or (name, visible_fraction). A visible fraction
    below 1 crops the photo below the head at that fraction of body height,
    like a stock photo cut at the knees, mid-thigh or waist."""
    if framing is None:
        framing = sample_framing(rng) if body_frac is None else ("full", 1.0)
    fname, vis = framing
    if out_w is None:
        lo, hi = (0.62, 0.82) if vis >= 0.999 else (0.68, 0.95)
        out_w = int(round(out_h * rng.uniform(lo, hi) / 2) * 2)
    W, H = out_w * ss, out_h * ss
    cx = W / 2 + rng.uniform(-0.08, 0.08) * W
    if vis >= 0.999:
        body_frac = body_frac or rng.uniform(0.70, 0.92)
        height = H * body_frac
        top = (H - height) * rng.uniform(0.25, 0.75)
    else:
        top = H * rng.uniform(0.025, 0.12)
        height = (H - top) / vis
    cv = _Canvas(W, H, cx, top, height)
    parts = _body_parts(spec, cv)

    skin = np.array(spec.skin, np.float32)
    rgb = np.zeros((H, W, 3), np.float32)
    body_union = np.maximum.reduce([parts["torso"], parts["head"], parts["arms"], parts["legs"]])
    rgb[body_union > 0] = skin
    # muscle / fold definition when shirtless
    if detail and spec.top == "none":
        line = (skin * 0.78).tolist()
        th = max(1, int(cv.r(0.0025)))
        if spec.label == "athletic":
            for y in (0.335, 0.375, 0.415):
                a, b = cv.p(-spec.W * 0.55, y), cv.p(spec.W * 0.55, y)
                cv2.line(rgb, tuple(a.astype(int)), tuple(b.astype(int)), line, th, cv2.LINE_AA)
            a, b = cv.p(0, 0.30), cv.p(0, 0.45)
            cv2.line(rgb, tuple(a.astype(int)), tuple(b.astype(int)), line, th, cv2.LINE_AA)
            for side in (-1, 1):
                c = cv.p(side * spec.C * 0.45, 0.265)
                cv2.ellipse(rgb, tuple(c.astype(int)), (int(cv.r(spec.C * 0.5)), int(cv.r(0.03))), 0, 10, 170, line, th, cv2.LINE_AA)
        elif spec.label == "obese":
            c = cv.p(0, 0.47)
            cv2.ellipse(rgb, tuple(c.astype(int)), (int(cv.r(spec.W * 0.8)), int(cv.r(0.02))), 0, 0, 180, line, th, cv2.LINE_AA)
        elif spec.label == "skinny":
            for y in (0.255, 0.275, 0.295):
                for side in (-1, 1):
                    a, b = cv.p(side * spec.C * 0.25, y), cv.p(side * spec.C * 0.7, y + 0.01)
                    cv2.line(rgb, tuple(a.astype(int)), tuple(b.astype(int)), (skin * 0.86).tolist(), th, cv2.LINE_AA)

    # clothing
    def fabric(colour, m):
        noise = rng.normal(0, 4, (H, W, 1)).astype(np.float32)
        rgb[m > 0] = (np.array(colour, np.float32) + noise[m > 0])
    fabric(spec.bottom_colour, parts["bottom"])
    fabric(spec.top_colour, parts["top"])
    rgb[parts["shoes"] > 0] = np.array(spec.shoe_colour, np.float32)

    # hair
    hair = np.zeros((H, W), np.uint8)
    hw = int(cv.r(0.049 * spec.head))
    hc = cv.p(0, 0.058)
    if spec.hair_style != "bald":
        cv2.ellipse(hair, tuple(hc.astype(int)), (hw, int(cv.r(0.058))), 0, 180, 360, 255, -1, cv2.LINE_AA)
        if spec.hair_style == "long":
            for side in (-1, 1):
                a = cv.p(side * 0.042 * spec.head, 0.06)
                b = cv.p(side * 0.05 * spec.head, 0.185)
                cv2.line(hair, tuple(a.astype(int)), tuple(b.astype(int)), 255, int(cv.r(0.024)), cv2.LINE_AA)
        if spec.hair_style == "bun":
            b = cv.p(0, 0.0)
            cv2.circle(hair, tuple(b.astype(int)), int(cv.r(0.022)), 255, -1, cv2.LINE_AA)
        rgb[hair > 0] = np.array(spec.hair, np.float32)
    sil = np.maximum(parts["silhouette"], hair)

    # face hint (eyes) for realism at larger render sizes
    if detail and out_h >= 400:
        for side in (-1, 1):
            e = cv.p(side * 0.017, 0.072)
            cv2.circle(rgb, tuple(e.astype(int)), max(1, int(cv.r(0.0045))), (40, 30, 30), -1, cv2.LINE_AA)
        m0, m1 = cv.p(-0.012, 0.100), cv.p(0.012, 0.100)
        cv2.line(rgb, tuple(m0.astype(int)), tuple(m1.astype(int)), (skin * 0.7).tolist(), max(1, int(cv.r(0.002))), cv2.LINE_AA)

    # volumetric shading: brighter in the middle of each limb, darker at edges
    dist = cv2.distanceTransform((sil > 127).astype(np.uint8), cv2.DIST_L2, 5)
    shade = 0.58 + 0.42 * (1 - np.exp(-dist / max(1.0, cv.r(0.011))))
    light_dir = rng.uniform(-1, 1)
    grad = 1 + 0.12 * light_dir * np.linspace(-1, 1, W)[None, :]
    rgb *= (shade * grad)[..., None]

    # slight body turn (3/4 view)
    if spec.turn:
        rgb, sil = _turn_warp(rgb, sil, cx, spec.turn)

    # small camera roll
    angle = rng.normal(0, 2.5)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    rgb = cv2.warpAffine(rgb, M, (W, H))
    sil = cv2.warpAffine(sil, M, (W, H))

    # downsample (anti-alias)
    rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(sil.astype(np.float32) / 255.0, (out_w, out_h), interpolation=cv2.INTER_AREA)

    if with_background:
        bg, bg_kind = _background(out_w, out_h, rng)
        # contact shadow (only when the feet are in frame)
        ys, xs = np.nonzero(alpha > 0.5)
        if len(ys) and vis >= 0.999:
            sh = np.zeros((out_h, out_w), np.float32)
            cv2.ellipse(sh, (int(xs.mean()), int(ys.max())), (int((xs.max() - xs.min()) * 0.45), max(2, out_h // 60)),
                        0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), out_h / 120)
            bg *= (1 - 0.45 * sh)[..., None]
        out = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
    else:
        bg_kind = "none"
        out = rgb * alpha[..., None] + 255 * (1 - alpha[..., None])

    # sensor effects
    out = out * rng.uniform(0.85, 1.12) + rng.uniform(-12, 12)
    out += rng.normal(0, rng.uniform(1, 5), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.9))
    if with_background and watermark and rng.random() < 0.15:
        out = _watermark(out, rng)
    mask = (alpha > 0.5).astype(np.uint8) * 255
    meta = spec.to_dict()
    meta.pop("arms_lr", None)
    meta["arm_left"], meta["arm_right"] = spec.arms_lr[0][0], spec.arms_lr[1][0]
    meta["framing"] = fname
    meta["visible"] = round(vis, 4)
    meta["turn"] = round(spec.turn, 3)
    meta["background"] = bg_kind
    meta["roll_deg"] = round(float(angle), 2)
    return out, mask, meta


def generate_person(label: str, seed: int, sex: str | None = None, pose: str | None = None, **kw) -> tuple:
    rng = np.random.default_rng(seed)
    spec = sample_body(label, rng, sex=sex, pose=pose)
    return render(spec, rng, **kw)
