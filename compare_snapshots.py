#!/usr/bin/env python3
"""Compare two Kokkos-style SWE snapshot CSV files (x,y,h)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    # names: x,y,h
    return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["h"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path, help="reference CSV (e.g. Kokkos)")
    p.add_argument("b", type=Path, help="candidate CSV (e.g. Taichi)")
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=1e-10)
    args = p.parse_args()

    ax, ay, ah = load_csv(args.a)
    bx, by, bh = load_csv(args.b)

    if ax.shape != bx.shape:
        raise SystemExit(f"Shape mismatch: a has {ax.shape}, b has {bx.shape}")

    # x/y should match exactly (they come from disk). Still report diffs to catch ordering issues.
    dx = np.abs(ax - bx)
    dy = np.abs(ay - by)
    dh = np.abs(ah - bh)

    def summarize(name: str, d: np.ndarray) -> str:
        return f"{name}: max={d.max():.3e}, mean={d.mean():.3e}, l2={np.linalg.norm(d):.3e}"

    print(summarize("diff(x)", dx))
    print(summarize("diff(y)", dy))
    print(summarize("diff(h)", dh))

    ok = np.allclose(ah, bh, rtol=args.rtol, atol=args.atol)
    if not ok:
        idx = int(np.argmax(dh))
        print(f"[FAIL] h mismatch at i={idx}: a={ah[idx]:.12g}, b={bh[idx]:.12g}, |diff|={dh[idx]:.3e}")
        raise SystemExit(2)
    print("[OK] h matches within tolerance")


if __name__ == "__main__":
    main()

