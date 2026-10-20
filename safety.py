"""Optional common local braking guard; never changes desired heading.

For each sensed pair at surface gap g, each agent moves at most 0.45*g.
The triangle inequality then leaves >= 0.1*g over the whole straight step,
irrespective of the two headings. Obstacles use 0.9*g. This is a conservative
discrete kinematic guard, not a claim about braking-limited physical vehicles.
"""
import numpy as np


def safe_step_lengths(pos, obstacles, config):
    delta = pos[:, None, :]-pos[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    sensed = (distance < config.sensing_radius) & ~np.eye(len(pos), dtype=bool)
    pair_gap = np.where(sensed, distance-2*config.agent_radius, np.inf)
    length = np.minimum(config.speed*config.dt, 0.45*np.maximum(pair_gap.min(axis=1), 0))
    for ox, oy, radius in obstacles:
        gap = np.linalg.norm(pos-[ox,oy],axis=1)-radius-config.agent_radius
        length = np.minimum(length, .9*np.maximum(gap,0))
    return length
