"""Deformed-grid labels -> reference-grid labels.

``warp.inverse_warp_gray`` renders def[x] = ref[x - d(x)]. The field d it is given is therefore indexed by the
DEFORMED pixel x: the point seen at x in the deformed image came from X = x - d(x). A network (RAFT's
coords1 - coords0) and subset DIC report, at every REFERENCE pixel X, where that point went:

    U(X) = x - X  with  x - d(x) = X,   i.e.   U(X) = d(X + U(X)).

The two agree for rigid motion and differ by about grad(d) . d otherwise (CONVENTION.md section 3).
``reference_grid_labels`` solves the fixed point; ``convert_dataset`` rewrites the labels of a dataset's
smooth samples and copies everything else byte for byte (the images are not touched).

Fields with a displacement jump are not converted here: a spline across the jump mixes the two sides. Mode II
sliding cracks (``crack_mode2_simple``) need no conversion, because each side moves parallel to the line and
rigidly, so U = d exactly; cracks with strain need a side-aware conversion.

Where the warp folds (det(I - grad d) <= 0), some reference material appears more than once in the deformed
image and no unique label exists. Those reference pixels, and any pixel the solver cannot bring below the
tolerance, are returned as not valid; ``convert_dataset`` writes them to the dataset's valid/ mask, which the
loader already applies to the loss (CONVENTION.md section 4).
"""
from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

SMOOTH_FIELD_TYPES = ("legacy_import", "smooth_random_peak")
TOL_PX = 1e-6
MAX_ITER = 100
MAX_MASKED_FRACTION = 0.05      # a sample with more unlabelled pixels than this stops the conversion
SUBSAMPLE = 5                   # sub-pixel samples per axis when mapping a fold region to the reference grid


def ambiguous_reference_pixels(u, v) -> np.ndarray:
    """Reference pixels with more than one match in the deformed image (bool, shape of u).

    The map m(x) = x - d(x) sends a deformed pixel to the reference point it shows. Where det(dm/dx) <= 0 the
    warp folds, and every reference point in m(fold region) is shown at several deformed positions. The fold
    region (grown by one pixel) is sampled at SUBSAMPLE x SUBSAMPLE points per pixel, mapped by m, rounded to
    the reference grid and grown by one pixel.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    fold = (1.0 - du_dx) * (1.0 - dv_dy) - du_dy * dv_dx <= 0
    H, W = u.shape
    out = np.zeros((H, W), dtype=bool)
    if not fold.any():
        return out
    ys, xs = np.nonzero(binary_dilation(fold))
    offs = (np.arange(SUBSAMPLE) + 0.5) / SUBSAMPLE - 0.5
    for oy in offs:
        for ox in offs:
            y, x = ys + oy, xs + ox
            X = np.rint(x - map_coordinates(u, [y, x], order=1, mode="nearest")).astype(int)
            Y = np.rint(y - map_coordinates(v, [y, x], order=1, mode="nearest")).astype(int)
            keep = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
            out[Y[keep], X[keep]] = True
    return binary_dilation(out)


def reference_grid_labels(u, v, *, order: int = 3, max_iter: int = MAX_ITER, tol: float = TOL_PX):
    """Solve U(X) = d(X + U(X)) for a deformed-grid field d = (u, v) of shape (H, W).

    d is interpolated with a spline of the given order (edge values extended outside the frame). The equation
    F(U) = U - d(X + U) = 0 is solved by Newton steps with the Jacobian I - grad d (finite differences of d,
    linearly interpolated; zero along an axis where x lies beyond the frame, as the extended d is flat there).
    A pixel whose Newton step does not lower its residual takes the plain fixed-point step U <- d(X + U)
    instead. Returns U, V (float64) and a bool mask of the pixels with a unique, solved label: final residual
    |F| below ``tol`` px, at a point x = X + U where the warp does not fold (det(I - grad d(x)) > 0), and not
    shown more than once in the deformed image (``ambiguous_reference_pixels``); other pixels come back False.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.ndim != 2 or u.shape != v.shape:
        raise ValueError(f"u and v must be 2-D arrays of the same shape, got {u.shape} and {v.shape}")
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise ValueError("u and v must be finite")
    du_dy, du_dx = np.gradient(u)
    dv_dy, dv_dx = np.gradient(v)
    H, W = u.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    def at(field, coords, k):
        # scipy pads by edge replication before the spline prefilter (one well-defined extension)
        return map_coordinates(field, coords, order=k, mode="nearest")

    def residual(U, V):
        coords = [yy + V, xx + U]
        ru = U - at(u, coords, order)
        rv = V - at(v, coords, order)
        return ru, rv, np.maximum(np.abs(ru), np.abs(rv))

    U, V = u.copy(), v.copy()                     # start from the deformed-grid value
    ru, rv, res = residual(U, V)
    for _ in range(max_iter):
        if res.max() < tol:
            break
        x, y = xx + U, yy + V
        coords = [y, x]
        in_x = ((x >= 0) & (x <= W - 1)).astype(np.float64)
        in_y = ((y >= 0) & (y <= H - 1)).astype(np.float64)
        a, b = 1.0 - in_x * at(du_dx, coords, 1), -in_y * at(du_dy, coords, 1)
        c, e = -in_x * at(dv_dx, coords, 1), 1.0 - in_y * at(dv_dy, coords, 1)
        det = a * e - b * c
        newton = det > 0.05
        safe_det = np.where(newton, det, 1.0)
        Un = U - np.where(newton, (e * ru - b * rv) / safe_det, ru)
        Vn = V - np.where(newton, (a * rv - c * ru) / safe_det, rv)
        _, _, res_n = residual(Un, Vn)
        worse = res_n >= res                      # fall back to the fixed-point step U <- d(X + U)
        U = np.where(worse, U - ru, Un)
        V = np.where(worse, V - rv, Vn)
        ru, rv, res = residual(U, V)
    jac = (1.0 - du_dx) * (1.0 - dv_dy) - du_dy * dv_dx      # det of d(x - d(x))/dx
    jac_at = at(jac, [yy + V, xx + U], 1)
    return U, V, (res < tol) & (jac_at > 0) & ~ambiguous_reference_pixels(u, v)


def _convert_one(args):
    """Worker: convert one sample -> (id, U, V, valid mask or None, stats).

    Raises if more than MAX_MASKED_FRACTION of the pixels have no unique solved label."""
    sid, u_path, v_path = args
    u0 = np.load(u_path)
    v0 = np.load(v_path)
    U, V, ok = reference_grid_labels(u0, v0)
    n_masked = int((~ok).sum())
    if n_masked > MAX_MASKED_FRACTION * ok.size:
        raise RuntimeError(f"sample {sid}: {n_masked} of {ok.size} pixels have no unique label (folding warp)")
    change = np.hypot(U - u0, V - v0)[ok]
    stats = {"median_change_px": float(np.median(change)), "p95_change_px": float(np.percentile(change, 95)),
             "max_change_px": float(change.max()), "n_masked": n_masked}
    return sid, U.astype(np.float32), V.astype(np.float32), (ok if n_masked else None), stats


def convert_dataset(src, dst, field_types=SMOOTH_FIELD_TYPES, workers: int = 1, log=print) -> dict:
    """Copy dataset ``src`` to ``dst`` with reference-grid labels for the samples of ``field_types``.

    Images, params.csv, split.json and valid/ are copied byte for byte; u/v of other field types too. The
    copy is built in ``<dst>.partial`` and renamed at the end, so an interrupted run never looks complete.
    Writes ``label_conversion.csv`` (per sample) and adds ``label_convention`` to dataset_meta.json.
    """
    src, dst = Path(src), Path(dst)
    field_types = tuple(field_types)
    bad = [t for t in field_types if t.startswith("crack")]
    if bad:
        raise ValueError(f"field types with a displacement jump need a side-aware conversion: {bad}")
    if dst.exists():
        raise FileExistsError(f"destination exists, refusing to overwrite: {dst}")
    for name in ("params.csv", "split.json", "ref", "def", "u", "v"):
        if not (src / name).exists():
            raise FileNotFoundError(f"{src} has no {name}")
    part = dst.with_name(dst.name + ".partial")
    if part.exists():
        raise FileExistsError(f"a previous partial copy exists, move it aside first: {part}")

    with open(src / "params.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    todo = [r["id"] for r in rows if r["field_type"] in field_types]
    keep = [r["id"] for r in rows if r["field_type"] not in field_types]
    log(f"[labels] {src.name}: converting {len(todo)} samples {field_types}, copying {len(keep)} unchanged")

    part.mkdir(parents=True)
    for name in ("params.csv", "split.json"):
        shutil.copy2(src / name, part / name)
    for d in ("ref", "def") + (("valid",) if (src / "valid").exists() else ()):
        shutil.copytree(src / d, part / d)
    (part / "u").mkdir()
    (part / "v").mkdir()
    for sid in keep:
        for c in ("u", "v"):
            shutil.copy2(src / c / f"{sid}_{c}.npy", part / c / f"{sid}_{c}.npy")

    jobs = [(sid, src / "u" / f"{sid}_u.npy", src / "v" / f"{sid}_v.npy") for sid in todo]
    stats = {}

    def store(sid, U, V, ok, st):
        np.save(part / "u" / f"{sid}_u.npy", U)
        np.save(part / "v" / f"{sid}_v.npy", V)
        if ok is not None:                        # pixels without a unique label: excluded from the loss
            (part / "valid").mkdir(exist_ok=True)
            vpath = part / "valid" / f"{sid}_valid.npy"
            if vpath.exists():
                ok = ok & np.load(vpath).astype(bool)
            np.save(vpath, ok)
        stats[sid] = st

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, item in enumerate(ex.map(_convert_one, jobs, chunksize=16), 1):
                store(*item)
                if k % 1000 == 0:
                    log(f"[labels]   {k}/{len(jobs)}")
    else:
        for item in map(_convert_one, jobs):
            store(*item)

    with open(part / "label_conversion.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "field_type", "converted", "median_change_px", "p95_change_px", "max_change_px",
                    "n_masked_pixels"])
        for r in rows:
            st = stats.get(r["id"])
            w.writerow([r["id"], r["field_type"], int(st is not None)]
                       + ([f"{st[k]:.6f}" for k in ("median_change_px", "p95_change_px", "max_change_px")]
                          + [st["n_masked"]] if st else ["", "", "", ""]))

    med = np.array([s["median_change_px"] for s in stats.values()]) if stats else np.zeros(1)
    mx = np.array([s["max_change_px"] for s in stats.values()]) if stats else np.zeros(1)
    summary = {
        "grid": "reference",
        "definition": "U(X) = d(X + U(X)) with def[x] = ref[x - d(x)]; U is where reference pixel X went",
        "method": f"Newton on U - d(X + U) = 0, cubic spline of d, edge values extended, residual < {TOL_PX} px",
        "change_stats_over": "labelled (valid) pixels of converted samples",
        "source_dataset": str(src),
        "converted_field_types": list(field_types),
        "unchanged_field_types": sorted({r["field_type"] for r in rows} - set(field_types)),
        "n_converted": len(stats),
        "n_unchanged": len(keep),
        "per_sample_median_change_px": {"median": float(np.median(med)), "max": float(med.max())},
        "per_sample_max_change_px": {"median": float(np.median(mx)), "max": float(mx.max())},
        "masked": {"rule": "no unique solved label (fold or residual >= tolerance): valid/ mask False",
                   "n_samples": sum(1 for st in stats.values() if st["n_masked"]),
                   "n_pixels": int(sum(st["n_masked"] for st in stats.values()))},
    }
    meta_path = src / "dataset_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["label_convention"] = summary
    if "n_valid_mask" in meta:                    # number of samples with a valid/ mask
        meta["n_valid_mask"] = len(list((part / "valid").glob("*_valid.npy"))) if (part / "valid").exists() else 0
    (part / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    part.rename(dst)
    log(f"[labels] done: {dst}")
    return summary
