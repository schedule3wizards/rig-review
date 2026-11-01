# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mattlock of Schedule 3 Wizards
"""rig_review.py <rigged.glb> [--profile humanoid|ogre] [--out-dir DIR] [--schema rig_schema.json]

Reviews a KayKit-armature rig (the 41-bone Knight skeleton fitted to a mesh by bind_kaykit.py) against the rules in
rig_schema.json and writes <name>_rig_review.json and <name>_rig_review.png (the review sheet).

Exit codes: 0 every rule passes, 1 at least one rule FAILs, 2 the input could not be read, 3 no rule FAILs but at least one
could not be measured (SKIP), 4 the review raised on a file that loaded (a defect in this script, not in the input).

numpy and Pillow only. The glb is parsed here: node hierarchy, the first skin, inverse bind matrices, POSITION, JOINTS_0,
WEIGHTS_0, indices and animations; the motion frames are linear blend skinning of the bind-pose vertices.

Frame convention (the binder's): Y up, +X is the figure's left (the .l bones), +Z forward, feet at y = 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8, 5120: np.int8, 5122: np.int16}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORM = {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}


class BadInput(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------- glb

def load_glb(path):
    try:
        data = open(path, "rb").read()
        if data[:4] != b"glTF":
            raise BadInput("not a glb (magic %r)" % data[:4])
        off = 12
        j = None
        binchunk = b""
        while off < len(data):
            ln, ty = struct.unpack("<II", data[off:off + 8])
            body = data[off + 8:off + 8 + ln]
            if ty == 0x4E4F534A:
                j = json.loads(body)
            elif ty == 0x004E4942 and not binchunk:
                binchunk = body
            off += 8 + ln
        if j is None:
            raise BadInput("no JSON chunk")
        return j, binchunk
    except BadInput:
        raise
    except Exception as e:  # noqa: BLE001
        raise BadInput("cannot read %s: %s" % (path, e))


def accessor(j, b, i, as_float=False):
    a = j["accessors"][i]
    if "bufferView" not in a:
        return np.zeros((a["count"], NC[a["type"]]), np.float32)
    v = j["bufferViews"][a["bufferView"]]
    dt = CT[a["componentType"]]
    n = NC[a["type"]]
    off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = v.get("byteStride", 0)
    item = np.dtype(dt).itemsize * n
    if stride and stride != item:
        raw = np.frombuffer(b, np.uint8, count=stride * a["count"], offset=off).reshape(a["count"], stride)[:, :item]
        out = raw.copy().view(dt).reshape(a["count"], n)
    else:
        out = np.frombuffer(b, dt, count=a["count"] * n, offset=off).reshape(a["count"], n)
    if as_float:
        out = out.astype(np.float64)
        if a.get("normalized") and a["componentType"] in NORM:
            out = out / NORM[a["componentType"]]
    return out


def quat_to_mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def trs_mat(t, r, s):
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(r) * np.asarray(s)[None, :]
    m[:3, 3] = t
    return m


def slerp(q0, q1, f):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = math.acos(min(1.0, d))
        q = (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


class Rig:
    """One glb: skeleton in the bind pose, the skinned vertices, and the clips."""

    def __init__(self, path):
        self.path = path
        j, b = load_glb(path)
        self.j, self.b = j, b
        if not j.get("skins"):
            raise BadInput("no skin: this glb is not a rig")
        nodes = j["nodes"]
        self.nodes = nodes
        self.parent = {}
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        skin = j["skins"][0]
        self.joint_nodes = list(skin["joints"])
        self.names = [nodes[n].get("name", str(n)) for n in self.joint_nodes]
        self.idx = {n: i for i, n in enumerate(self.names)}
        if "inverseBindMatrices" in skin:
            self.ibm = accessor(j, b, skin["inverseBindMatrices"], True).reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            self.ibm = np.stack([np.linalg.inv(self.node_world(n)) for n in self.joint_nodes])
        bind_world = np.linalg.inv(self.ibm)
        self.bind_world = bind_world
        self.head = {nm: bind_world[i][:3, 3].copy() for i, nm in enumerate(self.names)}
        self.joint_parent = {}
        for i, n in enumerate(self.joint_nodes):
            p = self.parent.get(n)
            while p is not None and p not in self.joint_nodes:
                p = self.parent.get(p)
            self.joint_parent[self.names[i]] = self.names[self.joint_nodes.index(p)] if p is not None else None
        self._load_mesh()
        self._load_anims()

    def node_world(self, n):
        m = np.eye(4)
        chain = []
        while n is not None:
            chain.append(n)
            n = self.parent.get(n)
        for k in reversed(chain):
            m = m @ self.local_of(k)
        return m

    def local_of(self, k, over=None):
        nd = self.nodes[k]
        if "matrix" in nd:
            return np.array(nd["matrix"], float).reshape(4, 4).T
        if over and k in over:
            t, r, s = over[k]
        else:
            t, r, s = None, None, None
        t = nd.get("translation", [0, 0, 0]) if t is None else t
        r = nd.get("rotation", [0, 0, 0, 1]) if r is None else r
        s = nd.get("scale", [1, 1, 1]) if s is None else s
        return trs_mat(t, r, s)

    def _load_mesh(self):
        j, b = self.j, self.b
        Vs, Js, Ws, Fs, part = [], [], [], [], []
        base = 0
        self.parts = []
        for ni, nd in enumerate(self.nodes):
            if "mesh" not in nd or nd.get("skin") != 0:
                continue
            for pi, p in enumerate(j["meshes"][nd["mesh"]]["primitives"]):
                at = p["attributes"]
                if "JOINTS_0" not in at or "WEIGHTS_0" not in at:
                    continue
                V = accessor(j, b, at["POSITION"], True)
                Jn = accessor(j, b, at["JOINTS_0"]).astype(np.int64)
                Wn = accessor(j, b, at["WEIGHTS_0"], True)
                if "indices" in p:
                    F = accessor(j, b, p["indices"]).astype(np.int64).reshape(-1)
                else:
                    F = np.arange(len(V))
                mode = p.get("mode", 4)
                F = F[:len(F) // 3 * 3].reshape(-1, 3) if mode == 4 else np.zeros((0, 3), np.int64)
                Vs.append(V)
                Js.append(Jn)
                Ws.append(Wn)
                Fs.append(F + base)
                self.parts.append((nd.get("name", str(ni)), base, base + len(V)))
                base += len(V)
        if not Vs:
            raise BadInput("no skinned mesh primitive with JOINTS_0 and WEIGHTS_0")
        self.V = np.concatenate(Vs)
        self.JI = np.concatenate(Js)
        self.W = np.concatenate(Ws)
        s = self.W.sum(1, keepdims=True)
        s[s == 0] = 1
        self.W = self.W / s
        self.F = np.concatenate(Fs)

    def _load_anims(self):
        j, b = self.j, self.b
        self.anims = {}
        for a in j.get("animations", []):
            chans = {}
            dur = 0.0
            for ch in a["channels"]:
                tg = ch["target"]
                if tg.get("path") not in ("translation", "rotation", "scale"):
                    continue
                sm = a["samplers"][ch["sampler"]]
                tt = accessor(j, b, sm["input"], True).reshape(-1)
                vv = accessor(j, b, sm["output"], True)
                if sm.get("interpolation") == "CUBICSPLINE":
                    vv = vv[1::3]
                chans.setdefault(tg["node"], {})[tg["path"]] = (tt, vv)
                if len(tt):
                    dur = max(dur, float(tt[-1]))
            self.anims[a.get("name", "")] = (chans, dur)

    def find_anim(self, *needles):
        for nd in needles:
            for k in self.anims:
                if k == nd:
                    return k
        low = {k.lower(): k for k in self.anims}
        for nd in needles:
            for k, v in low.items():
                if nd.lower() in k:
                    return v
        return None

    def pose_over(self, anim, t):
        chans, dur = self.anims[anim]
        over = {}
        for n, d in chans.items():
            t0, r0, s0 = None, None, None
            for path, (tt, vv) in d.items():
                if len(tt) == 0:
                    continue
                if t <= tt[0]:
                    val = vv[0]
                elif t >= tt[-1]:
                    val = vv[-1]
                else:
                    k = int(np.searchsorted(tt, t) - 1)
                    f = (t - tt[k]) / max(tt[k + 1] - tt[k], 1e-12)
                    val = slerp(vv[k], vv[k + 1], f) if path == "rotation" else vv[k] + f * (vv[k + 1] - vv[k])
                if path == "translation":
                    t0 = val
                elif path == "rotation":
                    r0 = val
                else:
                    s0 = val
            over[n] = (t0, r0, s0)
        return over

    def posed_matrices(self, anim=None, t=0.0):
        """Joint matrices (world * IBM) for a clip time, plus the world position of every joint."""
        over = self.pose_over(anim, t) if anim else None
        cache = {}

        def world(n):
            if n in cache:
                return cache[n]
            p = self.parent.get(n)
            m = self.local_of(n, over)
            cache[n] = (world(p) @ m) if p is not None else m
            return cache[n]
        jm = np.stack([world(n) @ self.ibm[i] for i, n in enumerate(self.joint_nodes)])
        wp = {nm: world(n)[:3, 3].copy() for nm, n in zip(self.names, self.joint_nodes)}
        return jm, wp

    def skin(self, jm, V=None):
        V = self.V if V is None else V
        out = np.zeros_like(V)
        for k in range(self.JI.shape[1]):
            w = self.W[:, k:k + 1]
            m = jm[self.JI[:, k]]
            out += w * (np.einsum("nij,nj->ni", m[:, :3, :3], V) + m[:, :3, 3])
        return out


# ------------------------------------------------------------------------------------------------------- mesh landmarks

def surface_points(V, F, n=250000, seed=0):
    """Area-weighted points on the triangles (the glb vertices alone are too sparse on a low-poly body)."""
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tot = area.sum()
    if tot <= 0 or len(F) == 0:
        return V.copy()
    k = rng.choice(len(F), size=n, p=area / tot)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1
    u = np.where(flip, 1 - u, u)
    v = np.where(flip, 1 - v, v)
    P = a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k])
    return np.concatenate([P, V])


def runs(mask, gap=0):
    """[(lo, hi)] index runs of True in a 1-D bool array, bridging gaps of up to `gap` cells."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    out = []
    lo = prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            out.append((lo, prev))
            lo = i
        prev = i
    out.append((lo, prev))
    return out


def nearest_run(vals, cell, target, gap=2):
    """The run of occupied `cell`-wide bins of `vals` that contains `target` (or is nearest to it): (lo, hi, distance)."""
    lo = vals.min()
    n = int((vals.max() - lo) / cell) + 1
    occ = np.zeros(n, bool)
    occ[((vals - lo) / cell).astype(int)] = True
    best = None
    for a, b in runs(occ, gap):
        ya, yb = lo + a * cell, lo + (b + 1) * cell
        d = 0.0 if ya <= target <= yb else min(abs(target - ya), abs(target - yb))
        if best is None or d < best[2]:
            best = (ya, yb, d)
    return best


class Landmarks:
    """Landmarks read off the bind-pose mesh alone (no bone positions steer them, so they can judge the bones).

    The bind pose is a T-pose: the arm is a roughly horizontal tube leaving the torso. For the side s (+1 = the figure's
    left, +X) the tube is tracked from the outermost surface point inward, slab by slab, until its vertical run merges into
    the torso (run height above 2 x the tube's own height): that slab is x_root, the torso edge at the arm.
    """

    def __init__(self, rig, P=None):
        V = rig.V
        self.H = float(V[:, 1].max() - V[:, 1].min())
        self.ymin = float(V[:, 1].min())
        self.ymax = float(V[:, 1].max())
        self.P = surface_points(V, rig.F) if P is None else P
        self.cell = 0.006 * self.H
        self.arm = {s: self._arm(s) for s in (1, -1)}
        # a prop fused to one hand (a shield, a sword) lengthens that side's tube: the arm is symmetric by design, so the
        # shorter side's tip and root stand for both
        a1, a2 = self.arm[1], self.arm[-1]
        yc = [self.root_centre(1), self.root_centre(-1)]
        self.asym = {"x_root": abs(a1["x_root"] - a2["x_root"]) / max(a1["x_root"], a2["x_root"], 1e-9),
                     "x_tip": abs(a1["xtip"] - a2["xtip"]) / max(a1["xtip"], a2["xtip"], 1e-9),
                     "tube_y_over_H": abs(yc[0][0] - yc[1][0]) / self.H if yc[0] and yc[1] else 1.0}
        self.reliable = self.asym["x_root"] < 0.15 and self.asym["x_tip"] < 0.08 and self.asym["tube_y_over_H"] < 0.03
        for k in ("xtip", "x_root"):
            m = min(self.arm[1][k], self.arm[-1][k])
            for s in (1, -1):
                self.arm[s][k] = m
        self._neck()

    def _arm(self, s):
        P, H, cell = self.P, self.H, self.cell
        x = s * P[:, 0]
        xtip = float(x.max())
        sel = x > xtip - 0.03 * H
        ytip = float(np.median(P[sel, 1]))
        slab = 0.012 * H
        trk = []
        ymid = ytip
        for x0 in np.arange(xtip - slab, 0.0, -slab):
            m = (x >= x0 - slab) & (x < x0)
            if m.sum() < 5:
                trk.append((x0, np.nan, np.nan, np.nan, np.nan))
                continue
            ya, yb, d = nearest_run(P[m, 1], cell, ymid)
            if d > 0.12 * H:
                trk.append((x0, np.nan, np.nan, np.nan, np.nan))
                continue
            zz = P[m & (P[:, 1] >= ya) & (P[:, 1] <= yb), 2]
            trk.append((x0, ya, yb, 0.5 * (zz.min() + zz.max()), zz.max() - zz.min()))
            ymid = 0.5 * (ya + yb)
        trk = np.array(trk, float)
        h = trk[:, 2] - trk[:, 1]
        n = len(trk)
        outer = h[: max(3, n // 2)]
        outer = outer[~np.isnan(outer)]
        tube = float(np.median(outer)) if len(outer) else float("nan")
        root_i = n - 1
        for i in range(n):
            if np.isnan(h[i]) or h[i] > 2.0 * tube:
                root_i = i
                break
        x_root = float(trk[max(root_i - 1, 0), 0])
        good = trk[:max(root_i, 1)]
        good = good[~np.isnan(good[:, 1])]
        return {"xtip": xtip, "ytip": ytip, "tube_h": tube, "x_root": x_root, "track": good}

    def centre_at(self, s, x):
        """Arm-tube centre (y, z) at |x| = x on side s, interpolated along the tracked tube; None outside the tube."""
        a = self.arm[s]
        if a is None or len(a["track"]) < 3:
            return None
        t = a["track"][::-1]
        if x < t[0, 0] or x > t[-1, 0]:
            return None
        return float(np.interp(x, t[:, 0], 0.5 * (t[:, 1] + t[:, 2]))), float(np.interp(x, t[:, 0], t[:, 3]))

    def root_centre(self, s):
        """Arm-tube centre (y, z) in the slab nearest the torso."""
        a = self.arm[s]
        t = a["track"]
        if len(t) == 0:
            return None
        r = t[-1]
        return float(0.5 * (r[1] + r[2])), float(r[3])

    def _neck(self):
        """neck_y: the highest level, scanning down from the crown, at which the central column (the silhouette run through
        x = 0) becomes at least 1.6 x the head's own width and stays so for 4 percent of H: the head's underside, where the
        neck, or with a cowl or a fur wrap the shoulder mass, begins. The head's own width is the median over the band
        4 to 12 percent of H below the crown."""
        P, H, cell = self.P, self.H, self.cell
        self.x_root = float(np.mean([a["x_root"] for a in self.arm.values()]))
        ys = np.arange(self.ymin, self.ymax + cell, cell)
        w = np.full(len(ys), np.nan)
        order = np.argsort(P[:, 1])
        py = P[order, 1]
        px = P[order, 0]
        lo_i = np.searchsorted(py, ys - 0.5 * cell)
        hi_i = np.searchsorted(py, ys + 0.5 * cell)
        for i in range(len(ys)):
            if hi_i[i] - lo_i[i] < 3:
                continue
            a, b, _ = nearest_run(px[lo_i[i]:hi_i[i]], cell, 0.0)
            w[i] = b - a
        self.ys, self.w = ys, w
        band = (ys <= self.ymax - 0.04 * H) & (ys >= self.ymax - 0.12 * H)
        self.w_head = float(np.nanmedian(w[band]))
        sustain = max(1, int(0.04 * H / cell))
        thr = 1.6 * self.w_head
        self.neck_y = None
        i = int(np.searchsorted(ys, self.ymax - 0.08 * H))
        while i - sustain >= 0:
            win = w[i - sustain:i + 1]
            if np.all(win >= thr):
                self.neck_y = float(ys[i])
                break
            i -= 1
        if self.neck_y is None:
            self.neck_y = float(self.ymin)
            self.neck_kind = "no level where the central column is 1.6 x the head width"
        else:
            self.neck_kind = "the column reaches 1.6 x the head width (%.2f m) here" % self.w_head
        self._crotch()

    def _crotch(self):
        """crotch_y: the highest level below 0.6 H at which the silhouette through x = 0 splits into two legs (a gap of at
        least 2 cells at x = 0 with material on both sides). None when the legs never separate (a skirt, a loincloth)."""
        P, H, cell = self.P, self.H, self.cell
        self.crotch_y = None
        for y0 in np.arange(self.ymin + 0.6 * H, self.ymin + 0.1 * H, -cell):
            m = np.abs(P[:, 1] - y0) < 0.5 * cell
            if m.sum() < 10:
                continue
            xs = P[m, 0]
            near = np.abs(xs) < 1.5 * cell
            if not near.any() and (xs < 0).any() and (xs > 0).any():
                self.crotch_y = float(y0)
                break


class Interior:
    """Filled occupancy of the bind-pose mesh on a voxel grid (pitch H/140). A voxel is inside when the wall (the surface
    samples, dilated by one voxel to seal pin holes) is met along every one of the six axis rays from it. numpy only."""

    def __init__(self, P, H):
        self.pitch = H / 140.0
        self.lo = P.min(0) - 3 * self.pitch
        n = np.ceil((P.max(0) + 3 * self.pitch - self.lo) / self.pitch).astype(int) + 1
        self.n = n
        g = np.zeros(tuple(n), bool)
        ij = ((P - self.lo) / self.pitch).astype(int)
        g[ij[:, 0], ij[:, 1], ij[:, 2]] = True
        w = g.copy()
        for ax in range(3):
            w |= np.roll(g, 1, ax) | np.roll(g, -1, ax)
        self.wall = w
        inside = np.ones_like(w)
        for ax in range(3):
            cs = np.cumsum(w, axis=ax) > 0
            rcs = np.flip(np.cumsum(np.flip(w, ax), axis=ax), ax) > 0
            inside &= cs & rcs
        self.inside = inside | w

    def contains(self, p):
        ij = np.floor((np.asarray(p) - self.lo) / self.pitch).astype(int)
        if np.any(ij < 0) or np.any(ij >= self.n):
            return False
        return bool(self.inside[tuple(ij)])



# ------------------------------------------------------------------------------------------------------------ the rules

DEFORM = ["hips", "spine", "chest", "head"] + [b + "." + s for s in "lr" for b in
                                              ("upperarm", "lowerarm", "wrist", "hand", "upperleg", "lowerleg", "foot", "toes")]
PAIRS = [(b + ".l", b + ".r") for b in ("upperarm", "lowerarm", "wrist", "hand", "handslot", "upperleg", "lowerleg", "foot", "toes")]
SIDE = {1: "l", -1: "r"}


class Unmeasured(Exception):
    pass


def seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def edge_list(F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    m = int(e.max()) + 1
    key = np.unique(e[:, 0].astype(np.int64) * m + e[:, 1])
    return np.stack([key // m, key % m], 1)


class Ctx:
    """Everything the rules read, computed once per rig."""

    def __init__(self, rig, profile, schema):
        self.rig, self.profile, self.schema = rig, profile, schema
        self.H = float(rig.V[:, 1].max() - rig.V[:, 1].min())
        self.h = rig.head
        self._L = self._I = self._M = self._PN = self._caps = None
        self.roles = roles(rig)
        self.rules = {r["id"]: r for r in schema["rules"]}

    def params(self, rid):
        p = dict(self.rules[rid]["params"])
        p.update(self.schema["profiles"][self.profile].get("overrides", {}).get(rid, {}))
        return p

    @property
    def L(self):
        if self._L is None:
            self._L = Landmarks(self.rig)
        return self._L

    @property
    def interior(self):
        if self._I is None:
            self._I = Interior(self.L.P, self.H)
        return self._I

    @property
    def motion(self):
        if self._M is None:
            self._M = motion_frames(self.rig)
        return self._M


def motion_frames(rig):
    """The sample frames: Idle mid, Walking_A swing extremes, the attack reach peak, the end of Death_A. Each is
    {label, anim, t, V (skinned vertices), wp (joint world positions)}."""
    out = []
    missing = []

    def add(label, anim, t):
        jm, wp = rig.posed_matrices(anim, t)
        out.append({"label": label, "anim": anim, "t": t, "V": rig.skin(jm), "wp": wp})

    idle = rig.find_anim("Idle")
    walk = rig.find_anim("Walking_A")
    atk = rig.find_anim("1H_Melee_Attack_Slice_Diagonal", "1H_Melee_Attack")
    death = rig.find_anim("Death_A", "death")
    for nm, a in (("Idle", idle), ("Walking_A", walk), ("1H attack", atk), ("Death_A", death)):
        if a is None:
            missing.append(nm)
    if idle:
        add("idle", idle, 0.5 * rig.anims[idle][1])
    if walk:
        dur = rig.anims[walk][1]
        ts = np.linspace(0, dur, 33)
        z = []
        for t in ts:
            _, wp = rig.posed_matrices(walk, t)
            z.append(wp["hand.l"][2] - wp["hips"][2])
        add("walk A", walk, float(ts[int(np.argmin(z))]))
        add("walk B", walk, float(ts[int(np.argmax(z))]))
    if atk:
        dur = rig.anims[atk][1]
        ts = np.linspace(0, dur, 25)
        best = (-1, 0)
        for t in ts:
            _, wp = rig.posed_matrices(atk, t)
            d = max(np.linalg.norm(wp["hand.r"] - wp["hips"]), np.linalg.norm(wp["hand.l"] - wp["hips"]))
            if d > best[0]:
                best = (d, t)
        add("attack peak", atk, float(best[1]))
    if death:
        add("death end", death, rig.anims[death][1])
    return {"frames": out, "missing": missing}


def _res(value, passed, detail=None, joints=None, note=""):
    return {"value": value, "pass": bool(passed), "measured": detail or {}, "joints": joints or [], "note": note}


def need(c, *names):
    for n in names:
        if n not in c.h:
            raise Unmeasured("joint %s not in the skin" % n)


def r_sym(c, p):
    need(c, "hips", "spine", "chest", "head", *[n for pr in PAIRS for n in pr])
    xm = float(np.mean([c.h[n][0] for n in ("hips", "spine", "chest", "head")]))
    worst, det, bad = 0.0, {}, []
    for l, r in PAIRS:
        pl, pr = c.h[l], c.h[r].copy()
        pr[0] = 2 * xm - pr[0]
        e = float(np.linalg.norm(pl - pr)) / c.H
        det[l.rsplit(".", 1)[0]] = round(e, 4)
        if e > p["max"]:
            bad += [l, r]
        worst = max(worst, e)
    det["midline_x"] = round(xm, 4)
    return _res(worst, worst <= p["max"], det, bad)


def r_shoulder_level(c, p):
    need(c, "upperarm.l", "upperarm.r")
    v = abs(c.h["upperarm.l"][1] - c.h["upperarm.r"][1]) / c.H
    return _res(v, v <= p["max"], {"yl": round(float(c.h["upperarm.l"][1]), 4), "yr": round(float(c.h["upperarm.r"][1]), 4)},
                ["upperarm.l", "upperarm.r"] if v > p["max"] else [])


def r_neck_gap(c, p):
    nb = c.roles["nb"]
    need(c, nb, "upperarm.l", "upperarm.r")
    ys = 0.5 * (c.h["upperarm.l"][1] + c.h["upperarm.r"][1])
    v = (c.h[nb][1] - ys) / c.H
    ok = p["min"] <= v <= p["max"]
    return _res(v, ok, {"neck_base_joint": nb, "neck_base_y": round(float(c.h[nb][1]), 4), "shoulder_y": round(float(ys), 4), "gap_m": round(float(c.h[nb][1] - ys), 4)},
                [nb, "upperarm.l", "upperarm.r"] if not ok else [])


def r_neck_mesh(c, p):
    nb = c.roles["nb"]
    need(c, nb)
    L = c.L
    v = (c.h[nb][1] - L.neck_y) / c.H
    ok = p["min"] <= v <= p["max"]
    return _res(v, ok, {"neck_y_mesh": round(L.neck_y, 4), "neck_base_joint": nb, "neck_base_y": round(float(c.h[nb][1]), 4)}, [nb] if not ok else [], L.neck_kind)


def r_skull(c, p):
    nb = c.roles["nb"]
    need(c, nb)
    L = c.L
    head_h = L.ymax - L.neck_y
    v = (L.ymax - c.h[nb][1]) / max(head_h, 1e-9)
    return _res(v, v >= p["min"], {"head_height_m": round(head_h, 4), "mesh_above_joint_m": round(L.ymax - float(c.h[nb][1]), 4)},
                [nb] if v < p["min"] else [], L.neck_kind)


def r_spine_order(c, p):
    need(c, "hips", "spine", "chest", "head")
    ch = ["hips", "spine", "chest", c.roles["nb"]]
    gaps = [(c.h[b][1] - c.h[a][1]) / c.H for a, b in zip(ch, ch[1:])]
    v = min(gaps)
    bad = []
    for (a, b), g in zip(zip(ch, ch[1:]), gaps):
        if g < p["min"]:
            bad += [a, b]
    return _res(v, v >= p["min"], {"gaps": [round(g, 4) for g in gaps]}, sorted(set(bad)))


def r_crotch(c, p):
    need(c, "hips")
    cy = c.L.crotch_y
    if cy is None:
        raise Unmeasured("the legs never separate in the silhouette (skirt or loincloth): no crotch landmark")
    v = (c.h["hips"][1] - cy) / c.H
    ok = p["min"] <= v <= p["max"]
    return _res(v, ok, {"crotch_y": round(cy, 4), "hips_y": round(float(c.h["hips"][1]), 4)}, ["hips"] if not ok else [])


def r_pivot_in(c, p):
    need(c, "upperarm.l", "upperarm.r")
    L = c.L
    det, bad, ratios = {}, [], []
    for s in (1, -1):
        a = L.arm[s]
        if len(a["track"]) < 3:
            raise Unmeasured("the %s arm tube could not be tracked" % SIDE[s])
        r = abs(c.h["upperarm." + SIDE[s]][0]) / a["x_root"]
        ratios.append(r)
        det[SIDE[s]] = {"ratio": round(r, 3), "x_ua": round(abs(float(c.h["upperarm." + SIDE[s]][0])), 4), "x_root": round(a["x_root"], 4)}
        if not p["min"] <= r <= p["max"]:
            bad.append("upperarm." + SIDE[s])
    worst = max(ratios, key=lambda r: max(r - p["max"], p["min"] - r))
    return _res(worst, not bad, det, bad)


def r_pivot_height(c, p):
    L = c.L
    det, bad, vs = {}, [], []
    for s in (1, -1):
        rc = L.root_centre(s)
        if rc is None:
            raise Unmeasured("the %s arm tube could not be tracked" % SIDE[s])
        v = (c.h["upperarm." + SIDE[s]][1] - rc[0]) / c.H
        vs.append(v)
        det[SIDE[s]] = {"dy_over_H": round(v, 4), "tube_centre_y": round(rc[0], 4)}
        if not p["min"] <= v <= p["max"]:
            bad.append("upperarm." + SIDE[s])
    return _res(max(vs, key=lambda v: max(v - p["max"], p["min"] - v)), not bad, det, bad)


def _chord(a, b, k):
    ab = b - a
    t = float((k - a) @ ab) / float(ab @ ab)
    off = float(np.linalg.norm(k - (a + t * ab)))
    return t, off


def r_elbow(c, p):
    det, bad, worst = {}, [], 0.0
    for s in "lr":
        need(c, "upperarm." + s, "lowerarm." + s, "wrist." + s)
        t, off = _chord(c.h["upperarm." + s], c.h["wrist." + s], c.h["lowerarm." + s])
        off /= c.H
        det[s] = {"frac": round(t, 3), "off_over_H": round(off, 4)}
        if not (p["frac_min"] <= t <= p["frac_max"] and off <= p["off_max"]):
            bad.append("lowerarm." + s)
        worst = max(worst, abs(t - 0.5), off)
    return _res(worst, not bad, det, bad)


def r_coverage(c, p):
    L = c.L
    det, bad, vs = {}, [], []
    for s in (1, -1):
        need(c, "upperarm." + SIDE[s], "handslot." + SIDE[s])
        v = (s * c.h["handslot." + SIDE[s]][0] - s * c.h["upperarm." + SIDE[s]][0]) / L.arm[s]["xtip"]
        vs.append(v)
        det[SIDE[s]] = {"coverage": round(v, 3), "x_tip": round(L.arm[s]["xtip"], 4)}
        if v < p["min"]:
            bad += ["upperarm." + SIDE[s], "handslot." + SIDE[s]]
    return _res(min(vs), not bad, det, bad)


def r_hand_at(c, p):
    L = c.L
    det, bad, vs = {}, [], []
    for s in (1, -1):
        need(c, "hand." + SIDE[s])
        v = (L.arm[s]["xtip"] - s * c.h["hand." + SIDE[s]][0]) / c.H
        vs.append(v)
        det[SIDE[s]] = {"tip_gap_over_H": round(v, 4)}
        if not p["min"] <= v <= p["max"]:
            bad.append("hand." + SIDE[s])
    worst = max(vs, key=lambda v: max(v - p["max"], p["min"] - v))
    return _res(worst, not bad, det, bad)


def r_on_axis(c, p):
    L = c.L
    det, bad, worst = {}, [], 0.0
    for s in (1, -1):
        for b in ("lowerarm", "wrist", "hand"):
            n = b + "." + SIDE[s]
            need(c, n)
            ctr = L.centre_at(s, abs(float(c.h[n][0])))
            if ctr is None:
                det[n] = "outside the tracked tube"
                continue
            d = max(abs(c.h[n][1] - ctr[0]), abs(c.h[n][2] - ctr[1])) / c.H
            det[n] = round(d, 4)
            worst = max(worst, d)
            if d > p["max"]:
                bad.append(n)
    return _res(worst, not bad, det, bad)


def r_knee(c, p):
    det, bad, worst = {}, [], 0.0
    for s in "lr":
        need(c, "upperleg." + s, "lowerleg." + s, "foot." + s)
        t, off = _chord(c.h["upperleg." + s], c.h["foot." + s], c.h["lowerleg." + s])
        off /= c.H
        det[s] = {"frac": round(t, 3), "off_over_H": round(off, 4)}
        if not (p["frac_min"] <= t <= p["frac_max"] and off <= p["off_max"]):
            bad.append("lowerleg." + s)
        worst = max(worst, abs(t - 0.5), off)
    return _res(worst, not bad, det, bad)


def r_ankle(c, p):
    ymin = c.L.ymin
    det, bad, vs = {}, [], []
    for s in "lr":
        need(c, "foot." + s)
        v = (c.h["foot." + s][1] - ymin) / c.H
        vs.append(v)
        det[s] = round(v, 4)
        if not p["min"] <= v <= p["max"]:
            bad.append("foot." + s)
    worst = max(vs, key=lambda v: max(v - p["max"], p["min"] - v))
    return _res(worst, not bad, det, bad)


def r_floor(c, p):
    ymin = c.L.ymin
    det, bad = {"mesh_ymin_over_H": round(ymin / c.H, 4)}, []
    worst = abs(ymin) / c.H
    if worst > p["floor_max"]:
        bad += ["foot.l", "foot.r"]
    for s in "lr":
        need(c, "toes." + s)
        v = (c.h["toes." + s][1] - ymin) / c.H
        det["toes_" + s] = round(v, 4)
        if not p["toes_min"] <= v <= p["toes_max"]:
            bad.append("toes." + s)
        worst = max(worst, v)
    return _res(worst, not bad, det, sorted(set(bad)))


def r_inside(c, p):
    I, P = c.interior, c.L.P
    det, bad, worst = {}, [], 0.0
    for n in DEFORM:
        need(c, n)
        q = c.h[n]
        if I.contains(q):
            continue
        d = float(np.min(np.linalg.norm(P - q, axis=1))) / c.H
        det[n] = round(d, 4)
        worst = max(worst, d)
        if d > p["max"]:
            bad.append(n)
    return _res(worst, not bad, det, bad)


def segments(rig, ymax):
    """Bone segments, each owned by the bone it starts at: a link from a joint to its child joint belongs to the parent.
    So chest owns chest->head, chest->upperarm.l and chest->upperarm.r (the shoulder girdle), hips owns hips->spine and the
    two hips->upperleg links, and every limb bone owns its link to the next joint. The head owns head->crown and the toes
    a short stub forward. Returns {bone: [(a, b), ...]}."""
    h = rig.head
    seg = {"hips": [(h["hips"], h["spine"]), (h["hips"], h["upperleg.l"]), (h["hips"], h["upperleg.r"])],
           "spine": [(h["spine"], h["chest"])],
           "chest": [(h["chest"], h["head"]), (h["chest"], h["upperarm.l"]), (h["chest"], h["upperarm.r"])],
           "head": [(h["head"], np.array([h["head"][0], ymax, h["head"][2]]))]}
    stub = np.array([0.0, 0.0, 0.05 * (ymax - rig.V[:, 1].min())])
    for s in "lr":
        seg["upperarm." + s] = [(h["upperarm." + s], h["lowerarm." + s])]
        seg["lowerarm." + s] = [(h["lowerarm." + s], h["wrist." + s])]
        seg["wrist." + s] = [(h["wrist." + s], h["hand." + s])]
        seg["hand." + s] = [(h["hand." + s], h["handslot." + s])]
        seg["upperleg." + s] = [(h["upperleg." + s], h["lowerleg." + s])]
        seg["lowerleg." + s] = [(h["lowerleg." + s], h["foot." + s])]
        seg["foot." + s] = [(h["foot." + s], h["toes." + s])]
        seg["toes." + s] = [(h["toes." + s], h["toes." + s] + stub)]
    return seg


LIMBS = {"arm.l": ["upperarm.l", "lowerarm.l", "wrist.l", "hand.l"], "arm.r": ["upperarm.r", "lowerarm.r", "wrist.r", "hand.r"],
         "leg.l": ["upperleg.l", "lowerleg.l", "foot.l", "toes.l"], "leg.r": ["upperleg.r", "lowerleg.r", "foot.r", "toes.r"]}
TORSO = ["hips", "spine", "chest"]


def limb_capture(rig, H, ymax, near, margin=0.0, min_frac=0.002, W=None, JI=None):
    """Weights against nearest bone, per bone. A vertex is clearly on a limb when that limb's bone segments are the nearest of
    the body's groups (left arm, right arm, left leg, right leg, torso, head), within near x H, and by at least margin x H
    nearer than the next group. Such a vertex is assigned to the limb bone whose segment is nearest. Per bone: the share of its
    vertices whose dominant weight group is not that limb. Bones with fewer than max(25, min_frac x vertex count) vertices are not judged.
    Returns {bone: (share, n_vertices)} and the mask of captured vertices."""
    W = rig.W if W is None else W
    JI = rig.JI if JI is None else JI
    seg = segments(rig, ymax)
    groups = dict(LIMBS)
    groups["torso"] = [b for b in TORSO if b in seg]
    groups["head"] = ["head"]
    gn = list(groups)
    Db = {b: np.min([seg_dist(rig.V, *e) for e in seg[b]], 0) for g in groups.values() for b in g if b in seg}
    D = np.stack([np.min([Db[b] for b in groups[g] if b in Db], 0) for g in gn], 1)
    gi = {b: gn.index(g) for g, bs in groups.items() for b in bs}
    gi["root"] = gn.index("torso")
    for s in "lr":  # handslot is the hand's mount: its weight is the hand's
        gi["handslot." + s] = gn.index("arm." + s)
    bg = np.array([gi.get(n, -1) for n in rig.names])
    n = len(rig.V)
    gw = np.zeros((n, len(gn)))
    for k in range(JI.shape[1]):
        idx = bg[JI[:, k]]
        for g in range(len(gn)):
            gw[:, g] += W[:, k] * (idx == g)
    dg, ng = gw.argmax(1), D.argmin(1)
    second = np.sort(D, 1)[:, 1]
    out, cap = {}, np.zeros(n, bool)
    for g, bones in LIMBS.items():
        i = gn.index(g)
        on = (ng == i) & (D[:, i] < near * H) & (second - D[:, i] > margin * H)
        stack = np.stack([Db[b] for b in bones], 1)
        nb = stack.argmin(1)
        for k, b in enumerate(bones):
            m = on & (nb == k)
            if m.sum() < max(25, min_frac * n):
                continue
            bad = m & (dg != i)
            cap |= bad
            out[b] = (float(bad.sum()) / int(m.sum()), int(m.sum()))
    return out, cap


def r_nearest(c, p):
    out, cap = limb_capture(c.rig, c.H, c.L.ymax, p["near"], p["margin"])
    if not out:
        raise Unmeasured("no limb bone has enough vertices clearly nearest to it")
    worst = max(v[0] for v in out.values())
    joints = [k for k, v in out.items() if v[0] > p["share_max"]]
    bad = joints
    return _res(worst, not bad, {k: {"share": round(v[0], 4), "vertices": v[1]} for k, v in out.items()}, joints)


def _side_group(rig):
    side = np.zeros(len(rig.names), int)
    grp = np.zeros(len(rig.names), int)  # 1 arm, 2 leg
    for i, n in enumerate(rig.names):
        if n.endswith(".l"):
            side[i] = 1
        elif n.endswith(".r"):
            side[i] = 2
        b = n.split(".")[0]
        if b in ("upperarm", "lowerarm", "wrist", "hand", "handslot"):
            grp[i] = 1
        elif b in ("upperleg", "lowerleg", "foot", "toes"):
            grp[i] = 2
    return side, grp


def r_lr_mix(c, p):
    rig = c.rig
    side, grp = _side_group(rig)
    limb = grp > 0
    wl = (rig.W * ((side[rig.JI] == 1) & limb[rig.JI])).sum(1)
    wr = (rig.W * ((side[rig.JI] == 2) & limb[rig.JI])).sum(1)
    bad = (wl > p["weight_min"]) & (wr > p["weight_min"])
    share = float(bad.sum()) / len(rig.V)
    return _res(share, share <= p["share_max"], {"mixed_vertices": int(bad.sum()), "of": len(rig.V)})


def r_arm_leg_mix(c, p):
    rig = c.rig
    side, grp = _side_group(rig)
    wa = (rig.W * (grp[rig.JI] == 1)).sum(1)
    wg = (rig.W * (grp[rig.JI] == 2)).sum(1)
    bad = (wa > p["weight_min"]) & (wg > p["weight_min"])
    share = float(bad.sum()) / len(rig.V)
    return _res(share, share <= p["share_max"], {"mixed_vertices": int(bad.sum()), "of": len(rig.V)})


def r_stretch(c, p):
    rig = c.rig
    M = c.motion
    if not M["frames"]:
        raise Unmeasured("no sample clip found (missing %s)" % ", ".join(M["missing"]))
    E = edge_list(rig.F)
    rest = np.linalg.norm(rig.V[E[:, 0]] - rig.V[E[:, 1]], axis=1)
    keep = rest > 0.004 * c.H
    E, rest = E[keep], rest[keep]
    det, worst = {}, 0.0
    for fr in M["frames"]:
        ln = np.linalg.norm(fr["V"][E[:, 0]] - fr["V"][E[:, 1]], axis=1)
        ratio = ln / rest
        share = float((ratio > p["max_ratio"]).mean())
        det[fr["label"]] = {"share": round(share, 5), "max_ratio": round(float(ratio.max()), 2)}
        worst = max(worst, share)
    if M["missing"]:
        det["missing_clips"] = M["missing"]
    return _res(worst, worst <= p["share_max"], det)


def r_swing(c, p):
    rig = c.rig
    walk = rig.find_anim("Walking_A")
    if walk is None:
        raise Unmeasured("no Walking_A clip")
    ts = np.linspace(0, rig.anims[walk][1], 33)
    rel = {"l": [], "r": []}
    for t in ts:
        _, wp = rig.posed_matrices(walk, t)
        for s in "lr":
            rel[s].append(wp["hand." + s][2] - wp["hips"][2])
    L = c.L
    det, bad, vs = {}, [], []
    for s, sd in (("l", 1), ("r", -1)):
        rng = (max(rel[s]) - min(rel[s])) / L.arm[sd]["xtip"]
        vs.append(rng)
        det[s] = round(float(rng), 3)
        if rng < p["min"]:
            bad.append("hand." + s)
    return _res(min(vs), not bad, det, bad)


CHECKS = {
    "sym.joint_pairs": r_sym, "shoulder.level": r_shoulder_level, "shoulder.neck_gap": r_neck_gap,
    "neck.at_mesh_neck": r_neck_mesh, "neck.skull_clear": r_skull, "spine.order": r_spine_order,
    "spine.hips_at_crotch": r_crotch, "arm.pivot_in_shoulder": r_pivot_in, "arm.pivot_height": r_pivot_height,
    "arm.elbow_mid": r_elbow, "arm.chain_coverage": r_coverage, "arm.hand_at_mesh_hand": r_hand_at, "arm.on_axis": r_on_axis,
    "leg.knee_mid": r_knee, "leg.ankle_height": r_ankle, "leg.feet_on_floor": r_floor, "joints.inside_mesh": r_inside,
    "weights.nearest_bone": r_nearest, "weights.no_lr_mix": r_lr_mix, "weights.no_arm_leg_mix": r_arm_leg_mix,
    "motion.edge_stretch": r_stretch, "motion.arm_swing_walk": r_swing,
}


# ------------------------------------------------------------------------------------ topology and placement

RE_CLAV = re.compile(r"clav|collar|shoulder", re.I)
RE_NECK = re.compile(r"^neck", re.I)


def roles(rig):
    """The expected tree (Hip -> Spine1..3 -> Neck -> Head; top spine -> clavicle L/R -> upper arm; Hip -> thigh L/R) matched
    through a name map. KayKit names: hips, spine, chest, head, upperarm, lowerarm, wrist, hand, upperleg, lowerleg, foot, toes.
    Clavicle = any joint named clav*, collar* or shoulder* that is the parent of an upperarm; neck = any joint named neck* between
    the head and the top spine joint. Returns {spine: [hips .. top spine joint], neck: [joints from the spine side up] , head,
    nb: the joint whose head is the neck base (the neck joint if there is one, else the head joint), clav: {l, r}, head_tail}."""
    jp = rig.joint_parent
    kids = {}
    for n, p in jp.items():
        kids.setdefault(p, []).append(n)
    r = {"head": "head" if "head" in rig.idx else None, "neck": [], "spine": [], "clav": {}, "head_tail": None}
    p = jp.get("head")
    while p is not None and RE_NECK.match(p):
        r["neck"].insert(0, p)
        p = jp.get(p)
    up = []
    while p is not None and p != "root":
        up.append(p)
        p = jp.get(p)
    r["spine"] = list(reversed(up))
    r["spine_top"] = r["spine"][-1] if r["spine"] else None
    r["nb"] = r["neck"][0] if r["neck"] else r["head"]
    for s in "lr":
        q = jp.get("upperarm." + s)
        r["clav"][s] = q if q and RE_CLAV.search(q) else None
    tails = [k for k in kids.get("head", []) if k in rig.head]
    if tails:
        r["head_tail"] = max(tails, key=lambda k: rig.head[k][1])
    return r


def _nb(c):
    return c.roles["nb"]


def r_spine_segments(c, p):
    sp = c.roles["spine"]
    n = len(sp)
    return _res(n, n >= p["min"], {"chain": sp + [c.roles["nb"]], "segments_hip_to_neck_base": n}, [] if n >= p["min"] else sp)


def r_clavicles(c, p):
    R = c.roles
    need(c, "upperarm.l", "upperarm.r")
    missing = [s for s in "lr" if R["clav"][s] is None]
    if missing:
        return _res(None, False, {"parent_of_upperarm": {s: c.rig.joint_parent.get("upperarm." + s) for s in "lr"}}, ["upperarm." + s for s in missing],
                    "no clavicle bone: the upper arm hangs from %s" % c.rig.joint_parent.get("upperarm.l"))
    det, bad, angs = {}, [], []
    nb = c.h[R["nb"]]
    for s in "lr":
        cl = R["clav"][s]
        a, b = c.h[cl], c.h["upperarm." + s]
        d = b - a
        ang = math.degrees(math.atan2(abs(d[1]), math.hypot(d[0], d[2])))
        off = float(np.linalg.norm(a - nb)) / c.H
        par = c.rig.joint_parent.get(cl)
        det[s] = {"angle_deg": round(ang, 1), "origin_off_neck_base_over_H": round(off, 4), "parent": par}
        angs.append(ang)
        if ang > p["angle_max"] or off > p["origin_max"] or par != R["spine_top"]:
            bad.append(cl)
    return _res(max(angs), not bad, det, bad)


def r_neck_bone(c, p):
    R = c.roles
    if not R["neck"]:
        return _res(None, False, {"head_parent": c.rig.joint_parent.get("head")}, ["head"],
                    "no neck bone: the head hangs from %s" % c.rig.joint_parent.get("head"))
    par = c.rig.joint_parent.get(R["neck"][0])
    ln = float(np.linalg.norm(c.h["head"] - c.h[R["neck"][0]])) / c.H
    ok = par == R["spine_top"] and p["min"] <= ln <= p["max"]
    return _res(ln, ok, {"neck_joint": R["neck"][0], "parent": par, "length_over_H": round(ln, 4)}, [] if ok else [R["neck"][0]])


def r_head_span(c, p):
    R = c.roles
    need(c, "head")
    L = c.L
    f = (c.h["head"][1] - L.neck_y) / max(L.ymax - L.neck_y, 1e-9)
    if R["head_tail"] is None:
        return _res(None, False, {"head_joint_in_head_height": round(f, 3)}, ["head"],
                    "the head joint has no child joint: the bone's tail (the crown) is not stored, so it cannot span the skull")
    gap = abs(c.h[R["head_tail"]][1] - L.ymax) / c.H
    ok = gap <= p["max"] and f <= p["head_base_max"]
    return _res(gap, ok, {"tail": R["head_tail"], "tail_y": round(float(c.h[R["head_tail"]][1]), 4), "crown_y": round(L.ymax, 4),
                          "head_joint_in_head_height": round(f, 3)}, [] if ok else ["head", R["head_tail"]])


def r_chest_rib(c, p):
    need(c, "hips", "chest", "upperarm.l", "upperarm.r")
    sh = 0.5 * (c.h["upperarm.l"][1] + c.h["upperarm.r"][1])
    hy = c.h["hips"][1]
    fr = {n: (c.h[n][1] - hy) / (sh - hy) for n in c.roles["spine"] if n != "hips"}
    mid = 0.5 * (p["min"] + p["max"])
    best = min(fr, key=lambda n: abs(fr[n] - mid))
    ok = p["min"] <= fr[best] <= p["max"]
    return _res(fr[best], ok, {"fractions_hips_to_shoulder_line": {n: round(v, 3) for n, v in fr.items()}, "nearest_to_rib_base": best},
                [] if ok else [best])


def r_hip_sockets(c, p):
    L = c.L
    det, bad = {}, []
    worst = 0.0
    for s in "lr":
        need(c, "upperleg." + s)
        q = c.h["upperleg." + s]
        lat = abs(q[0]) / c.H
        if L.crotch_y is None:
            raise Unmeasured("the legs never separate in the silhouette: no crotch height")
        hh = (q[1] - L.crotch_y) / c.H
        det[s] = {"lateral_over_H": round(lat, 4), "above_crotch_over_H": round(hh, 4)}
        if not (p["lat_min"] <= lat <= p["lat_max"] and p["h_min"] <= hh <= p["h_max"]):
            bad.append("upperleg." + s)
        worst = max(worst, max(p["lat_min"] - lat, lat - p["lat_max"], p["h_min"] - hh, hh - p["h_max"], 0.0))
    return _res(worst, not bad, det, bad)


def _section_centre(P, s, y, xj, H):
    m = (np.abs(P[:, 1] - y) < 0.01 * H) & (s * P[:, 0] > 0.01 * H) & (np.abs(P[:, 0] - xj) < 0.15 * H)
    if m.sum() < 20:
        return None
    x = P[m, 0]
    z = P[m, 2]
    return 0.5 * (np.percentile(x, 3) + np.percentile(x, 97)), 0.5 * (np.percentile(z, 3) + np.percentile(z, 97))


def r_knee_ankle(c, p):
    P = c.L.P
    det, bad, worst = {}, [], 0.0
    for s, sd in (("l", 1), ("r", -1)):
        for b in ("lowerleg", "foot"):
            n = b + "." + s
            need(c, n)
            q = c.h[n]
            ctr = _section_centre(P, sd, q[1], q[0], c.H)
            if ctr is None:
                det[n] = "no mesh section at this height"
                continue
            d = max(abs(q[0] - ctr[0]), abs(q[2] - ctr[1])) / c.H
            det[n] = round(d, 4)
            worst = max(worst, d)
            if d > p["max"]:
                bad.append(n)
    return _res(worst, not bad, det, bad)


CHECKS.update({"topology.spine_segments": r_spine_segments, "topology.clavicles": r_clavicles, "topology.neck_bone": r_neck_bone,
               "topology.head_span": r_head_span, "placement.chest_rib_base": r_chest_rib, "placement.hip_sockets": r_hip_sockets,
               "placement.knee_ankle_on_mesh": r_knee_ankle})


# ------------------------------------------------- arm pivot, upper/fore arm, idle clearance, deltoid volume, hip height

def surface_points_n(V, F, n=300000, seed=0):
    """Area-weighted surface points with the face normal of each."""
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    k = rng.choice(len(F), size=n, p=area / area.sum())
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    flip = (u + v) > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    P = a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k])
    N = cr[k] / np.maximum(np.linalg.norm(cr[k], axis=1, keepdims=True), 1e-12)
    return P, N


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    s = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    return s[:3], float(np.sqrt(max(s[3] + s[:3] @ s[:3], 0.0)))


def deltoid_cap(c, s):
    """The centre of the sphere fitted to the deltoid cap on side s (+1 left, -1 right). Starting inside the arm root (0.8 x_root
    out, a little above the tube centre), repeat 12 times: take the surface points in a shell 0.5 to 1.6 radii around the current
    centre whose normals face away from it, fit a sphere (radius clamped to 0.04 to 0.10 H) and move halfway to its centre."""
    L, H = c.L, c.H
    if c._PN is None:
        c._PN = surface_points_n(c.rig.V, c.rig.F)
    P, N = c._PN
    P, N = P * [s, 1, 1], N * [s, 1, 1]
    rc = L.root_centre(s)
    if rc is None:
        return None
    O = np.array([0.8 * L.x_root, rc[0] + 0.04 * H, rc[1]])
    r0 = 0.09 * H
    n = 0
    for _ in range(12):
        d = P - O
        dist = np.linalg.norm(d, axis=1)
        m = (dist < 1.6 * r0) & (dist > 0.5 * r0)
        if m.sum() < 80:
            return None
        u = d[m] / dist[m, None]
        Nn = N[m] * np.sign((N[m] * u).sum(1))[:, None]
        Q = P[m][(Nn * u).sum(1) > 0.8]
        if len(Q) < 60:
            return None
        cc, r = fit_sphere(Q)
        r0 = float(np.clip(r, 0.04 * H, 0.10 * H))
        O = 0.5 * O + 0.5 * cc
        n = len(Q)
    return {"centre": O * [s, 1, 1], "radius": r0, "points": n}


def _caps(c):
    if c._caps is None:
        c._caps = {s: deltoid_cap(c, sd) for s, sd in (("l", 1), ("r", -1))}
    return c._caps


def r_glenohumeral(c, p):
    caps = _caps(c)
    if caps["l"] is None or caps["r"] is None:
        raise Unmeasured("no deltoid cap could be fitted on one side")
    cl, cr = caps["l"]["centre"], caps["r"]["centre"].copy()
    cr[0] = -cr[0]
    dis = float(np.linalg.norm(cl - cr)) / c.H
    if dis > p["fit_agree_max"]:
        raise Unmeasured("the left and right cap fits disagree by %.3f H (limit %.3f): the fit is not reliable on this mesh" % (dis, p["fit_agree_max"]))
    det, bad, ds = {"left_right_fit_disagreement_over_H": round(dis, 4)}, [], []
    for s in "lr":
        need(c, "upperarm." + s)
        d = float(np.linalg.norm(caps[s]["centre"] - c.h["upperarm." + s])) / c.H
        ds.append(d)
        det[s] = {"distance_over_H": round(d, 4), "cap_centre": [round(float(x), 3) for x in caps[s]["centre"]], "cap_radius": round(caps[s]["radius"], 3)}
        if d > p["max"]:
            bad.append("upperarm." + s)
    return _res(max(ds), not bad, det, bad)


def mesh_elbow_x(c, s):
    """|x| of the narrowest section of the arm tube between the bicep and the forearm: smoothed tube height over 3 percent of H,
    searched from 10 to 60 percent of the way from the torso edge to the fingertip."""
    a = c.L.arm[s]
    t = a["track"][::-1]
    if len(t) < 8:
        return None
    x = t[:, 0]
    h = t[:, 2] - t[:, 1]
    k = max(1, int(0.03 * c.H / (x[1] - x[0] if len(x) > 1 else 0.01)))
    hs = np.convolve(h, np.ones(2 * k + 1) / (2 * k + 1), "same")
    L = a["xtip"] - a["x_root"]
    sel = (x > a["x_root"] + 0.10 * L) & (x < a["x_root"] + 0.60 * L)
    if sel.sum() < 3:
        return None
    xs, hh = x[sel], hs[sel]
    j = int(np.argmin(hh))
    if j == 0 or j == len(xs) - 1:
        return None  # the minimum sits at the edge of the window: no clear narrowing between bicep and forearm
    return float(xs[j])


def r_upper_fore(c, p):
    det, bad, rs = {}, [], []
    for s, sd in (("l", 1), ("r", -1)):
        need(c, "upperarm." + s, "lowerarm." + s, "wrist." + s)
        up = float(np.linalg.norm(c.h["lowerarm." + s] - c.h["upperarm." + s]))
        fo = float(np.linalg.norm(c.h["wrist." + s] - c.h["lowerarm." + s]))
        ratio = up / fo
        me = mesh_elbow_x(c, sd)
        off = None if me is None else abs(abs(float(c.h["lowerarm." + s][0])) - me) / c.H
        det[s] = {"upper_over_fore": round(ratio, 3), "elbow_off_mesh_elbow_over_H": None if off is None else round(off, 4), "mesh_elbow_x": None if me is None else round(me, 3)}
        rs.append(ratio)
        if not p["min"] <= ratio <= p["max"] or (off is not None and off > p["elbow_max"]):
            bad.append("lowerarm." + s)
    worst = max(rs, key=lambda r: max(r - p["max"], p["min"] - r))
    return _res(worst, not bad, det, bad)


def r_upper_len_mesh(c, p):
    caps = _caps(c)
    det, bad, rs = {}, [], []
    for s, sd in (("l", 1), ("r", -1)):
        if caps[s] is None:
            raise Unmeasured("no deltoid cap fitted")
        me = mesh_elbow_x(c, sd)
        if me is None:
            raise Unmeasured("no mesh elbow found")
        need(c, "upperarm." + s, "lowerarm." + s)
        up = float(np.linalg.norm(c.h["lowerarm." + s] - c.h["upperarm." + s]))
        hum = me - abs(float(caps[s]["centre"][0]))
        r = up / hum
        rs.append(r)
        det[s] = {"rig_upper_arm": round(up, 3), "mesh_humerus_cap_to_elbow": round(hum, 3), "ratio": round(r, 3)}
        if not p["min"] <= r <= p["max"]:
            bad.append("upperarm." + s)
    worst = max(rs, key=lambda r: max(r - p["max"], p["min"] - r))
    return _res(worst, not bad, det, bad)


def _torso_mask(rig):
    dom = rig.JI[np.arange(len(rig.V)), rig.W.argmax(1)]
    ids = [i for i, n in enumerate(rig.names) if n in ("hips", "root", "neck", "head", "chest") or n.startswith(("spine", "upper_chest"))]
    return np.isin(dom, ids)


def _elbow_clearance(rig, tm, anim, t, H):
    """Posed frame: elbow offset beyond the torso side, over H (negative = the elbow is inside the torso silhouette). The torso side
    is the 95th percentile of the lateral offset of torso-dominant vertices within 4 percent of H of the elbow's height along the
    spine axis."""
    jm, wp = rig.posed_matrices(anim, t)
    Vp = rig.skin(jm)
    o = wp["hips"]
    ax = wp["chest"] - wp["hips"]
    ax = ax / np.linalg.norm(ax)
    side = wp["upperarm.l"] - wp["upperarm.r"]
    side = side - ax * (side @ ax)
    side = side / np.linalg.norm(side)
    h_all = (Vp - o) @ ax
    out = []
    for s, sg in (("l", 1), ("r", -1)):
        q = wp["lowerarm." + s]
        band = tm & (np.abs(h_all - (q - o) @ ax) < 0.04 * H)
        if band.sum() < 10:
            continue
        hw = np.percentile(sg * ((Vp[band] - o) @ side), 95)
        out.append((sg * ((q - o) @ side) - hw) / H)
    return out


def r_idle_abduction(c, p):
    rig = c.rig
    idle, walk = rig.find_anim("Idle"), rig.find_anim("Walking_A")
    if idle is None or walk is None:
        raise Unmeasured("no Idle or Walking_A clip")
    tm = _torso_mask(rig)
    idle_v = [v for t in (0.0, 0.5 * rig.anims[idle][1]) for v in _elbow_clearance(rig, tm, idle, t, c.H)]
    walk_v = [v for t in np.linspace(0, rig.anims[walk][1], 9)[:-1] for v in _elbow_clearance(rig, tm, walk, t, c.H)]
    if not idle_v or not walk_v:
        raise Unmeasured("no torso vertices near the elbow")
    jm, wp = rig.posed_matrices(idle, 0.0)
    d = wp["lowerarm.l"] - wp["upperarm.l"]
    ax = wp["chest"] - wp["hips"]
    ax = ax / np.linalg.norm(ax)
    lat = d - ax * (d @ ax)
    ang = float(np.degrees(np.arctan2(abs(lat[0]), abs(d @ ax))))
    hi, lo = max(idle_v), min(walk_v)
    ok = hi <= p["idle_max"] and lo >= p["walk_min"]
    return _res(hi, ok, {"idle_elbow_clearance_over_H": [round(float(v), 3) for v in idle_v], "walk_min_clearance_over_H": round(float(lo), 3),
                         "walk_mean_clearance_over_H": round(float(np.mean(walk_v)), 3),
                         "upper_arm_angle_off_torso_deg_in_the_clip": round(ang, 1)}, [] if ok else ["upperarm.l", "upperarm.r", "lowerarm.l", "lowerarm.r"],
                "the upper-arm angle is set by the clip's rotations and the T-pose bind, so it is the same for every rig bound from this armature")


def _region(c):
    """The deltoid and proximal bicep, by position and not by weights (so a bad weighting cannot remove its own evidence): bind-pose
    vertices from the upperarm head out to 60 percent of the upper arm, within 0.10 H of its axis."""
    rig, H = c.rig, c.H
    m = np.zeros(len(rig.V), bool)
    for s in "lr":
        ua, la = rig.head["upperarm." + s], rig.head["lowerarm." + s]
        ax = la - ua
        ln = float(np.linalg.norm(ax))
        ax = ax / ln
        d = rig.V - ua
        t = (d @ ax) / ln
        perp = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1)
        m |= (t >= 0.0) & (t < 0.6) & (perp < 0.10 * H)
    return m


def r_deltoid_volume(c, p):
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        raise Unmeasured("scipy is not installed: the convex-hull volume cannot be computed")
    rig = c.rig
    m = _region(c)
    if m.sum() < 40:
        raise Unmeasured("fewer than 40 deltoid/bicep vertices")
    v0 = ConvexHull(rig.V[m]).volume
    det, worst = {}, 1.0
    for lab, names in (("chop", ("1H_Melee_Attack_Chop",)), ("slice", ("1H_Melee_Attack_Slice_Horizontal", "1H_Melee_Attack_Slice_Diagonal")), ("idle", ("Idle",))):
        an = rig.find_anim(*names)
        if an is None:
            continue
        rs = []
        for t in np.linspace(0, rig.anims[an][1], 9):
            jm, _ = rig.posed_matrices(an, t)
            rs.append(ConvexHull(rig.skin(jm)[m]).volume / v0)
        det[lab] = {"min": round(float(min(rs)), 3), "max": round(float(max(rs)), 3)}
        worst = max(worst, max(rs), 1.0 / max(min(rs), 1e-6))
    if not det:
        raise Unmeasured("no chop, slice or idle clip in the glb")
    ok = all(p["min"] <= v["min"] and v["max"] <= p["max"] for v in det.values())
    return _res(worst, ok, det, [] if ok else ["upperarm.l", "upperarm.r"])


def r_hips_height(c, p):
    L = c.L
    if L.crotch_y is None:
        raise Unmeasured("the legs never separate in the silhouette: no crotch height")
    need(c, "hips", "upperleg.l", "upperleg.r")
    det, bad, worst = {}, [], 0.0
    hy = c.h["hips"][1]
    for s in "lr":
        ul = c.h["upperleg." + s][1]
        above = (ul - L.crotch_y) / c.H
        pelvis = (hy - ul) / c.H
        det[s] = {"thigh_joint_above_crotch_over_H": round(above, 4), "hips_above_thigh_joint_over_H": round(pelvis, 4)}
        if not p["thigh_min"] <= above <= p["thigh_max"]:
            bad.append("upperleg." + s)
        if not p["hips_min"] <= pelvis <= p["hips_max"]:
            bad.append("hips")
        worst = max(worst, p["thigh_min"] - above, above - p["thigh_max"], p["hips_min"] - pelvis, pelvis - p["hips_max"])
    return _res(worst, not bad, det, sorted(set(bad)))


CHECKS.update({"arm.pivot_glenohumeral": r_glenohumeral, "arm.upper_fore_ratio": r_upper_fore, "arm.upper_length_vs_mesh": r_upper_len_mesh,
               "motion.idle_arm_abduction": r_idle_abduction, "motion.deltoid_volume": r_deltoid_volume, "placement.hips_height": r_hips_height})


def threshold_text(rule, p):
    return "%s  [%s]" % (rule["pass_when"], ", ".join("%s=%s" % kv for kv in p.items()))


def load_schema(path=None):
    path = path or os.path.join(HERE, "rig_schema.json")
    try:
        s = json.load(open(path, encoding="utf8"))
    except Exception as e:  # noqa: BLE001
        raise BadInput("cannot read schema %s: %s" % (path, e))
    missing = [r["id"] for r in s["rules"] if r["id"] not in CHECKS]
    if missing:
        raise BadInput("schema rules with no check in rig_review.py: %s" % missing)
    return s


def analyse(rig, profile="humanoid", schema=None, only=None):
    schema = schema or load_schema()
    if profile not in schema["profiles"]:
        raise BadInput("unknown profile %s (have %s)" % (profile, list(schema["profiles"])))
    c = Ctx(rig, profile, schema)
    results = []
    for rule in schema["rules"]:
        if only and rule["id"] not in only:
            continue
        p = c.params(rule["id"])
        base = {"id": rule["id"], "group": rule["group"], "threshold": threshold_text(rule, p)}
        try:
            r = CHECKS[rule["id"]](c, p)
            base.update(status="PASS" if r["pass"] else "FAIL", value=None if r["value"] is None else round(float(r["value"]), 5),
                        measured=r["measured"], joints=r["joints"], note=r["note"])
        except Unmeasured as e:
            base.update(status="SKIP", value=None, measured={}, joints=[], note=str(e))
        except (KeyError, ValueError, IndexError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            base.update(status="SKIP", value=None, measured={}, joints=[], note="could not measure: %s: %s" % (type(e).__name__, e))
        results.append(base)
    return c, results


# ----------------------------------------------------------------------------------------------------- the review sheet

def verdict_of(n):
    """FAIL when any rule fails; INCOMPLETE when none fails and any was skipped; PASS only when every rule passed."""
    return "FAIL" if n["FAIL"] else ("INCOMPLETE" if n["SKIP"] else "PASS")


def chains_for(R):
    """Bone chains to draw: the spine to the head (through any neck and head tail), each arm from the clavicle or the top spine
    joint, each leg from the hips."""
    sp = R["spine"] + R["neck"] + ["head"] + ([R["head_tail"]] if R["head_tail"] else [])
    out = [sp]
    for s in "lr":
        cl = R["clav"][s]
        out.append(([R["spine_top"]] + ([cl] if cl else [])) + [b + "." + s for b in ("upperarm", "lowerarm", "wrist", "hand", "handslot")])
        out.append(["hips"] + [b + "." + s for b in ("upperleg", "lowerleg", "foot", "toes")])
    return out


COL = {"PASS": (30, 130, 50), "FAIL": (205, 30, 30), "SKIP": (130, 130, 130), "INCOMPLETE": (200, 100, 0)}


def _font(sz):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


def draw_overlay(d, box, view, rig, c, results, ft):
    """One bind-pose overlay: mesh points, bones, guide lines (neck base, shoulder line, hips, mesh neck), failing joints."""
    x0, y0, x1, y1 = box
    V = rig.V
    ax = 0 if view == "front" else 2
    sgn = 1.0
    ys_lo, ys_hi = V[:, 1].min(), V[:, 1].max()
    a_lo, a_hi = V[:, ax].min(), V[:, ax].max()
    for q in rig.head.values():
        a_lo, a_hi = min(a_lo, q[ax]), max(a_hi, q[ax])
    pad = 24
    sc = min((x1 - x0 - 2 * pad) / max(a_hi - a_lo, 1e-6), (y1 - y0 - 2 * pad - 14) / max(ys_hi - ys_lo, 1e-6))
    cx = 0.5 * (a_lo + a_hi)

    def px(p):
        return (0.5 * (x0 + x1) + sgn * (p[ax] - cx) * sc, y1 - pad - (p[1] - ys_lo) * sc)
    d.rectangle(box, outline=(200, 200, 200))
    pts = V[:: max(1, len(V) // 30000)]
    for p in pts:
        d.point(px(p), fill=(150, 160, 190))
    h = rig.head
    guides = []
    nbj = c.roles["nb"]
    if nbj in h:
        guides.append(("neck base", h[nbj][1], (200, 0, 200)))
    if "upperarm.l" in h and "upperarm.r" in h:
        guides.append(("shoulders", 0.5 * (h["upperarm.l"][1] + h["upperarm.r"][1]), (230, 120, 0)))
    if "hips" in h:
        guides.append(("hips", h["hips"][1], (0, 150, 60)))
    for lab, y, col in guides:
        yy = y1 - pad - (y - ys_lo) * sc
        d.line([(x0 + 2, yy), (x1 - 2, yy)], fill=col, width=1)
        d.text((x0 + 4, yy - 13), lab, fill=col, font=ft)
    if c._L is not None:
        yy = y1 - pad - (c.L.neck_y - ys_lo) * sc
        for xx in range(int(x0), int(x1), 10):
            d.line([(xx, yy), (xx + 4, yy)], fill=(60, 120, 220), width=1)
        d.text((x1 - 70, yy - 13), "mesh neck", fill=(60, 120, 220), font=ft)
    for ch in chains_for(c.roles):
        seg = [h[n] for n in ch if n in h]
        for a, b in zip(seg, seg[1:]):
            d.line([px(a), px(b)], fill=(220, 30, 30), width=2)
        for q in seg:
            xx, yy = px(q)
            d.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=(20, 20, 200))
    bad = {}
    for r in results:
        if r["status"] == "FAIL":
            for j in r["joints"]:
                bad.setdefault(j, []).append(r["id"])
    k = 0
    for j, ids in bad.items():
        if j not in h:
            continue
        xx, yy = px(h[j])
        d.ellipse((xx - 9, yy - 9, xx + 9, yy + 9), outline=(255, 0, 0), width=3)
        d.text((xx + 11, yy - 6 + (k % 3) * 11), j, fill=(200, 0, 0), font=ft)
        k += 1
    d.text((x0 + 6, y0 + 4), view + " (bind pose)", fill=(0, 0, 0), font=ft)


def render_poses(rig, c, ft, tile_w, tile_h):
    """The pose strip from one fixed 3/4 camera: bind, idle, two walk extremes, attack peak, death end."""
    from PIL import Image, ImageDraw
    M = c.motion
    frames = [("bind", rig.V, rig.head)] + [(f["label"], f["V"], f["wp"]) for f in M["frames"]]
    yaw, pitch = math.radians(35), math.radians(14)
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R = Rx @ Ry
    proj = [(lab, V @ R.T, wp) for lab, V, wp in frames]
    allp = np.concatenate([p for _, p, _ in proj])
    lo, hi = allp.min(0), allp.max(0)
    # death poses lie on the floor: the common frame comes from the bind pose and the walk, not from the death pose alone
    keep = np.concatenate([p for lab, p, _ in proj if lab != "death end"])
    lo2, hi2 = keep.min(0), keep.max(0)
    lo = np.minimum(lo2, np.maximum(lo, lo2 - 0.5 * (hi2 - lo2)))
    hi = np.maximum(hi2, np.minimum(hi, hi2 + 0.3 * (hi2 - lo2)))
    sc = min((tile_w - 20) / max(hi[0] - lo[0], 1e-6), (tile_h - 36) / max(hi[1] - lo[1], 1e-6))
    light = np.array([-0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    F = rig.F
    if len(F) > 90000:
        F = F[:: len(F) // 90000 + 1]
    imgs = []
    for lab, P, wp in proj:
        im = Image.new("RGB", (tile_w, tile_h), (246, 246, 248))
        d = ImageDraw.Draw(im)
        sx = 10 + (P[:, 0] - lo[0]) * sc
        sy = tile_h - 14 - (P[:, 1] - lo[1]) * sc
        a, b, cc = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
        n = np.cross(b - a, cc - a)
        ln = np.linalg.norm(n, axis=1)
        ln[ln == 0] = 1
        n = n / ln[:, None]
        sh = 0.40 + 0.60 * np.abs(n @ light)
        order = np.argsort((a[:, 2] + b[:, 2] + cc[:, 2]))
        for t in order:
            i0, i1, i2 = F[t]
            g = sh[t]
            d.polygon([(sx[i0], sy[i0]), (sx[i1], sy[i1]), (sx[i2], sy[i2])], fill=(int(165 * g), int(172 * g), int(195 * g)))
        for ch in chains_for(c.roles):
            seg = [(wp[nm] @ R.T) for nm in ch if nm in wp]
            pp = [(10 + (q[0] - lo[0]) * sc, tile_h - 14 - (q[1] - lo[1]) * sc) for q in seg]
            d.line(pp, fill=(220, 30, 30), width=2)
            for q in pp:
                d.ellipse((q[0] - 2.5, q[1] - 2.5, q[0] + 2.5, q[1] + 2.5), fill=(20, 20, 200))
        d.text((6, 4), lab, fill=(0, 0, 0), font=ft)
        imgs.append(im)
    return imgs


def short_threshold(r):
    t = r["threshold"]
    i = t.find("[")
    return t[i + 1:-1].replace("=", " ").replace(", ", "  ") if i >= 0 else t


def render_sheet(rig, c, results, name, profile, out_png):
    from PIL import Image, ImageDraw
    ft, fb = _font(13), _font(17)
    TW, TH = 350, 400
    groups = sum(1 for i, r in enumerate(results) if i == 0 or r["group"] != results[i - 1]["group"])
    table_end = 54 + groups * 15 + len(results) * 17
    W, topH = 6 * TW, max(780, table_end + 34)
    img = Image.new("RGB", (W, topH + TH + 34), (255, 255, 255))
    d = ImageDraw.Draw(img)
    n = {k: sum(1 for r in results if r["status"] == k) for k in ("PASS", "FAIL", "SKIP")}
    verdict = verdict_of(n)
    d.text((10, 8), "%s   profile=%s   %d FAIL  %d PASS  %d SKIP   %s" % (name, profile, n["FAIL"], n["PASS"], n["SKIP"], verdict),
           fill=COL[verdict], font=fb)
    try:
        if not c.L.reliable:
            d.text((900, 10), "LANDMARKS UNRELIABLE: the two arms read differently (%s); mesh-relative rules may be wrong"
                   % ", ".join("%s %.2f" % kv for kv in c.L.asym.items()), fill=(200, 100, 0), font=ft)
    except Exception:  # noqa: BLE001
        pass
    draw_overlay(d, (10, 34, 600, topH - 6), "front", rig, c, results, ft)
    draw_overlay(d, (610, 34, 1200, topH - 6), "side", rig, c, results, ft)
    y = 38
    d.text((1215, y - 4), "rule", fill=(0, 0, 0), font=ft)
    d.text((1465, y - 4), "value", fill=(0, 0, 0), font=ft)
    d.text((1535, y - 4), "limit", fill=(0, 0, 0), font=ft)
    y += 16
    last = None
    for r in results:
        if r["group"] != last:
            last = r["group"]
            d.text((1215, y), last.upper(), fill=(90, 90, 90), font=ft)
            y += 15
        col = COL[r["status"]]
        d.rectangle((1215, y + 2, 1226, y + 12), fill=col)
        d.text((1232, y), r["id"], fill=col, font=ft)
        v = "-" if r["value"] is None else ("%.4f" % r["value"])
        d.text((1465, y), v, fill=col, font=ft)
        d.text((1535, y), short_threshold(r)[:62] if r["status"] != "SKIP" else r["note"][:46], fill=(70, 70, 70), font=ft)
        y += 17
    d.text((1215, topH - 24), "bind pose overlays: red ring = joint named by a failing rule; magenta = neck base,", fill=(90, 90, 90), font=ft)
    d.text((1215, topH - 10), "orange = shoulder line, green = hips, dashed blue = neck level read off the mesh", fill=(90, 90, 90), font=ft)
    try:
        tiles = render_poses(rig, c, ft, TW, TH)
        for i, t in enumerate(tiles[:6]):
            img.paste(t, (i * TW, topH + 14))
        d.text((10, topH - 4 + 4), "pose strip, one fixed 3/4 camera (35 deg yaw, 14 deg pitch), linear blend skinning of the glb", fill=(0, 0, 0), font=ft)
    except Exception as e:  # noqa: BLE001
        d.text((10, topH + 20), "pose strip unavailable: %s" % e, fill=(200, 0, 0), font=ft)
    img.save(out_png)


# ----------------------------------------------------------------------------------------------------------------- cli

def landmarks_summary(c):
    try:
        L = c.L
        return {"H": round(L.H, 4), "ymin": round(L.ymin, 4), "ymax": round(L.ymax, 4), "neck_y": round(L.neck_y, 4), "head_width": round(L.w_head, 4),
                "crotch_y": None if L.crotch_y is None else round(L.crotch_y, 4), "x_root": round(L.x_root, 4),
                "x_tip_l": round(L.arm[1]["xtip"], 4), "x_tip_r": round(L.arm[-1]["xtip"], 4),
                "x_root_l": round(L.arm[1]["x_root"], 4), "x_root_r": round(L.arm[-1]["x_root"], 4),
                "left_right_disagreement": {k: round(v, 3) for k, v in L.asym.items()}, "landmarks_reliable": bool(L.reliable)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def review(path, profile="humanoid", out_dir=None, name=None, schema=None, sheet=True, rig=None):
    rig = rig or Rig(path)
    schema = schema or load_schema()
    c, results = analyse(rig, profile, schema)
    name = name or default_name(path)
    n = {k: sum(1 for r in results if r["status"] == k) for k in ("PASS", "FAIL", "SKIP")}
    doc = {"file": path, "name": name, "profile": profile, "schema": schema["schema"],
           "summary": {"pass": n["PASS"], "fail": n["FAIL"], "skip": n["SKIP"], "verdict": verdict_of(n)},
           "failed": [r["id"] for r in results if r["status"] == "FAIL"], "landmarks": landmarks_summary(c), "rules": results}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, name + "_rig_review.json"), "w", encoding="utf8") as f:
            json.dump(doc, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))
        if sheet:
            render_sheet(rig, c, results, name, profile, os.path.join(out_dir, name + "_rig_review.png"))
    return doc


def default_name(path):
    p = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(p))[0]
    parent = os.path.basename(os.path.dirname(p))
    return parent + "__" + stem if parent else stem


def main(argv=None):
    ap = argparse.ArgumentParser(description="Review a KayKit-armature rig against rig_schema.json.")
    ap.add_argument("glb")
    ap.add_argument("--profile", default="humanoid")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "rig_reviews"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args(argv)
    try:
        schema = load_schema(a.schema)
        rig = Rig(a.glb)
    except BadInput as e:
        print("rig_review: broken input: %s" % e, file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print("rig_review: broken input: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    try:
        doc = review(a.glb, a.profile, a.out_dir, a.name, schema, sheet=not a.no_sheet, rig=rig)
    except BadInput as e:
        print("rig_review: broken input: %s" % e, file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("rig_review: internal error: the review raised on a file that loaded; this is a defect in rig_review.py",
              file=sys.stderr)
        return 4
    s = doc["summary"]
    for r in doc["rules"]:
        print("%-4s %-26s %-10s %s" % (r["status"], r["id"], "" if r["value"] is None else r["value"], r["note"][:70] if r["status"] == "SKIP" else ""))
    print("%s profile=%s: %d FAIL, %d PASS, %d SKIP -> %s" % (doc["name"], a.profile, s["fail"], s["pass"], s["skip"], s["verdict"]))
    if s["fail"]:
        print("failed: " + ", ".join(doc["failed"]))
    elif s["skip"]:
        print("skipped: " + ", ".join(r["id"] for r in doc["rules"] if r["status"] == "SKIP"))
    return 1 if s["fail"] else (3 if s["skip"] else 0)


if __name__ == "__main__":
    sys.exit(main())
