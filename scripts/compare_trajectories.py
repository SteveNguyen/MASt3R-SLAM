#!/usr/bin/env python3
"""Side-by-side trajectory comparison with Sim(3) Umeyama alignment.

Reads two grabette-format CSVs (frame_idx, timestamp, is_lost, x, y, z, q_*),
aligns the primary onto the reference using Sim(3) (or SE(3) / no alignment),
and emits matplotlib plots: 3D overlay, per-axis time series, and the
translation residual.

Run from the project root, inside the docker container or any env that has
matplotlib + pandas:

    uv run python scripts/compare_trajectories.py \\
        --primary   test_data/grabette9/mast3r_camera_trajectory.csv \\
        --reference test_data/grabette9/mapping_camera_trajectory.csv \\
        --output    test_data/grabette9/comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) (or SE(3) when with_scale=False) alignment such
    that  s * R @ src.T + t  ≈ dst.T.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = len(src)
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    sc, dc = src - src_mean, dst - dst_mean
    H = sc.T @ dc / n
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    if with_scale:
        var_src = (sc ** 2).sum() / n
        s = (S * np.diag(D)).sum() / var_src
    else:
        s = 1.0
    t = dst_mean - s * R @ src_mean
    return float(s), R, t


def load_traj(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["is_lost"].astype(bool)].reset_index(drop=True)
    return df


def equal_aspect_3d(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(0), points.max(0)
    half = (maxs - mins).max() / 2
    centers = (mins + maxs) / 2
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, type=Path,
                    help="Trajectory to align ONTO the reference (e.g. MASt3R-SLAM)")
    ap.add_argument("--reference", required=True, type=Path,
                    help="Reference trajectory (e.g. ORB-SLAM3, Quest, mocap)")
    ap.add_argument("--align", choices=("sim3", "se3", "none"), default="sim3",
                    help="sim3 = scale+rotation+translation (default); "
                         "se3 = rigid (rotation+translation, no scale); "
                         "none = raw")
    ap.add_argument("--output", type=Path, default=None,
                    help="Directory to save PNGs. If omitted, plt.show().")
    ap.add_argument("--primary-name", default=None,
                    help="Display name for primary (default: filename stem)")
    ap.add_argument("--reference-name", default=None,
                    help="Display name for reference (default: filename stem)")
    args = ap.parse_args()

    pri_name = args.primary_name or args.primary.stem
    ref_name = args.reference_name or args.reference.stem

    pri = load_traj(args.primary)
    ref = load_traj(args.reference)

    common = pd.merge(
        pri[["frame_idx", "timestamp", "x", "y", "z"]],
        ref[["frame_idx", "timestamp", "x", "y", "z"]],
        on="frame_idx",
        suffixes=("_pri", "_ref"),
    )
    print(f"primary tracked frames:   {len(pri):>5}  ({args.primary})")
    print(f"reference tracked frames: {len(ref):>5}  ({args.reference})")
    print(f"common frames:            {len(common):>5}")
    if len(common) < 3:
        raise SystemExit("Need >= 3 common frames for alignment")

    src = common[["x_pri", "y_pri", "z_pri"]].to_numpy()
    dst = common[["x_ref", "y_ref", "z_ref"]].to_numpy()

    s, R, t = umeyama(src, dst, with_scale=(args.align == "sim3"))
    print(f"\nalignment ({args.align}):")
    print(f"  scale: {s:.4f}")
    print(f"  R:\n{R}")
    print(f"  t: {t}")

    # Apply transform to ALL primary frames (not just common ones)
    pri_xyz = pri[["x", "y", "z"]].to_numpy()
    pri_xyz_aligned = (s * (R @ pri_xyz.T)).T + t

    # Per-frame error at common timestamps
    src_aligned = (s * (R @ src.T)).T + t
    err = np.linalg.norm(src_aligned - dst, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"\ntranslation residual at common frames (m):")
    print(f"  rmse:   {rmse:.4f}")
    print(f"  mean:   {err.mean():.4f}")
    print(f"  median: {np.median(err):.4f}")
    print(f"  max:    {err.max():.4f}")
    print(f"  std:    {err.std():.4f}")

    pri_path = float(np.linalg.norm(np.diff(pri_xyz, axis=0), axis=1).sum())
    ref_path = float(np.linalg.norm(
        np.diff(ref[["x", "y", "z"]].to_numpy(), axis=0), axis=1).sum())
    print(f"\npath length:  {pri_name}={pri_path:.3f} m   {ref_name}={ref_path:.3f} m")
    print(f"path ratio:   {pri_path/ref_path:.4f}")

    # ---------------------------------------------------------------
    # Figure 1: 3D overlay + per-axis vs time
    # ---------------------------------------------------------------
    ref_xyz = ref[["x", "y", "z"]].to_numpy()
    fig = plt.figure(figsize=(14, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(*ref_xyz.T, color="C2", linewidth=1.4, label=f"{ref_name} (ref)")
    ax3d.plot(*pri_xyz_aligned.T, color="C1", linewidth=1.4, alpha=0.9,
              label=f"{pri_name} (aligned)")
    ax3d.scatter(*ref_xyz[0], color="green", s=50, marker="o", label="ref start")
    ax3d.scatter(*ref_xyz[-1], color="red", s=50, marker="s", label="ref end")
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    ax3d.legend(loc="best", fontsize=8)
    ax3d.set_title(f"3D trajectory (align={args.align}, scale={s:.3f})")
    equal_aspect_3d(ax3d, np.vstack([ref_xyz, pri_xyz_aligned]))

    for k, name in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(2, 2, k + 2)
        ax.plot(ref["timestamp"], ref[name], color="C2", linewidth=1.0,
                label=ref_name)
        ax.plot(pri["timestamp"], pri_xyz_aligned[:, k], color="C1",
                linewidth=1.0, alpha=0.9, label=f"{pri_name} (aligned)")
        ax.set_xlabel("time [s]"); ax.set_ylabel(f"{name} [m]")
        ax.set_title(f"{name}(t)"); ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    # ---------------------------------------------------------------
    # Figure 2: residual error
    # ---------------------------------------------------------------
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(common["timestamp_ref"], err, color="C3", linewidth=1.0)
    axes[0].set_xlabel("time [s]"); axes[0].set_ylabel("|residual| [m]")
    axes[0].set_title(f"translation residual at common frames "
                      f"(rmse={rmse:.3f} m)")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(err, bins=30, color="C3", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("|residual| [m]"); axes[1].set_ylabel("count")
    axes[1].set_title("residual distribution")
    axes[1].grid(True, alpha=0.3)
    fig2.tight_layout()

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output / "trajectories.png", dpi=120,
                    bbox_inches="tight")
        fig2.savefig(args.output / "residuals.png", dpi=120,
                     bbox_inches="tight")
        print(f"\nsaved plots to {args.output}/")
    else:
        plt.show()


if __name__ == "__main__":
    main()
