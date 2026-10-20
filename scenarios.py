"""Fixed geometry, shared across controllers. Migration is along positive y."""
import numpy as np
from .config import SCENARIOS


def lattice(n, spacing):
    width = int(np.ceil(np.sqrt(n)))
    xy = np.array([(i % width, i // width) for i in range(n)], dtype=float) * spacing
    return xy - xy.mean(axis=0)


def make_scenario(config, name, seed):
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    n = config.n
    groups = np.zeros(n, dtype=int)
    if name == "split":
        half = n // 2
        left = lattice(half, config.initial_spacing)
        right = lattice(n-half, config.initial_spacing)
        left += rng.uniform(-config.initial_jitter, config.initial_jitter, left.shape)
        right += rng.uniform(-config.initial_jitter, config.initial_jitter, right.shape)
        left[:, 0] -= left[:, 0].max() + config.split_gap/2
        right[:, 0] += config.split_gap/2 - right[:, 0].min()
        pos = np.vstack([left, right])
        groups[half:] = 1
        theta = np.pi/2 + np.where(groups == 0, config.split_angle, -config.split_angle)
    else:
        pos = lattice(n, config.initial_spacing)
        pos += rng.uniform(-config.initial_jitter, config.initial_jitter, pos.shape)
        groups = (pos[:, 0] >= 0).astype(int)
        theta = np.full(n, np.pi/2)
    theta += rng.normal(0, 0.015, n)
    # Keep first obstacle ahead of the initial flock for all N.
    y0 = float(lattice(n, config.initial_spacing)[:, 1].max() + config.obstacle_first_offset)
    r, spacing = config.obstacle_radius, config.obstacle_row_spacing
    layouts = {
        "split": [], "open": [],
        "single": [(0.0, y0, r)],
        "double": [(0.0, y0, r), (-2.0, y0+spacing, r)],
        "triple": [(0.0, y0, r), (-2.0, y0+spacing, r), (2.0, y0+2*spacing, r)],
    }
    obstacles = np.array(layouts[name], dtype=float).reshape(-1, 3)
    return pos, theta, groups, obstacles
