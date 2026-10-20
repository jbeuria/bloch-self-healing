"""Local controller adapted from code/swarm_nonperiodic.py.

2D, degree-normalized alignment; bounded turns. Bloch channels use exact
dissipation / rotation / dissipation with the field frozen during each step.
This preserves the unit ball when t2 <= 2*t1, without component clipping.
"""
import numpy as np
from .config import MODES
from .safety import safe_step_lengths


def bloch_step(m, hz, gamma, t1, t2, dt):
    q = np.tanh(hz)
    out = m.copy()
    def damp():
        out[..., :2] *= np.exp(-dt/(2*t2))
        out[..., 2] = q + (out[..., 2]-q)*np.exp(-dt/(2*t1))
    damp()
    h = np.zeros_like(out)
    h[..., 0], h[..., 2] = gamma, hz
    norm = np.linalg.norm(h, axis=-1)
    axis = h / np.maximum(norm[..., None], 1e-15)
    angle = 2*norm*dt
    c, s = np.cos(angle)[..., None], np.sin(angle)[..., None]
    rotated = (out*c + np.cross(axis, out)*s
               + axis*np.sum(axis*out, axis=-1)[..., None]*(1-c))
    out[...] = rotated
    damp()
    return out


def geometry(pos, radius):
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    adj = (dist < radius) & ~np.eye(len(pos), dtype=bool)
    return delta, dist, adj


def stationary_z(hz, gamma, t1, t2):
    """Continuous Bloch fixed-field stationary z, including transverse drive.

    This matches the DC response, not the full transient/frequency response.
    """
    return np.tanh(hz)/(1+4*gamma**2*t1*t2/(1+4*hz**2*t2**2))


class Controller:
    def __init__(self, config, mode):
        if mode not in MODES:
            raise ValueError(f"unknown controller {mode}")
        self.c, self.mode = config, mode
        self.m = np.zeros((config.n, config.n, 3))
        self.s = np.full(config.n, config.slow_base, dtype=float)
        if mode == "frozen":
            self.s[:] = config.frozen_tone

    def heading_command(self, pos, theta, obstacles, noise):
        """Update internal states and return the requested heading, before actuation."""
        c = self.c
        delta, dist, adj = geometry(pos, c.sensing_radius)
        degree = adj.sum(axis=1)
        heading = np.column_stack([np.cos(theta), np.sin(theta)])
        use_feedback = self.mode in ("full", "scalar", "frozen", "scalar_matched", "instant_matched")
        hz = c.g_align*(heading @ heading.T)
        if use_feedback:
            hz += c.kappa*self.s[:, None]
        if self.mode == "reactive":
            z = np.tanh(hz)
        elif self.mode == "instant_matched":
            z = stationary_z(hz, c.gamma, c.t1, c.t2)
        else:
            old = self.m
            if self.mode in ("scalar", "scalar_matched"):
                new = np.zeros_like(old)
                q = (stationary_z(hz, c.gamma, c.t1, c.t2)
                     if self.mode == "scalar_matched" else np.tanh(hz))
                new[..., 2] = q + (old[..., 2]-q)*np.exp(-c.dt/c.t1)
            else:
                new = bloch_step(old, hz, c.gamma, c.t1, c.t2, c.dt)
            if c.channel_policy == "reset":
                self.m = np.where(adj[..., None], new, 0.0)
            elif c.channel_policy == "retain":
                self.m = np.where(adj[..., None], new, old)
            else:
                # Decay inactive channels toward zero without observing lost neighbors.
                decay = np.array([np.exp(-c.dt/c.t2)]*2+[np.exp(-c.dt/c.t1)])
                self.m = np.where(adj[..., None], new, old*decay)
            z = self.m[..., 2]
        weights = adj * (1+z)/2
        # Divide by degree, not total weight: state still changes alignment strength.
        align = (weights @ heading) / np.maximum(degree[:, None], 1)
        cohesion = -(adj[..., None]*delta).sum(axis=1)/np.maximum(degree[:, None], 1)
        unit = delta / np.maximum(dist[..., None], 1e-12)
        sep_weight = np.where(adj, np.maximum(1-dist/c.separation_radius, 0), 0)
        separation = (sep_weight[..., None]*unit).sum(axis=1)
        avoid = np.zeros_like(pos)
        for ox, oy, rad in obstacles:
            rel = pos - [ox, oy]
            d = np.linalg.norm(rel, axis=1)
            strength = np.clip(1-(d-rad-c.agent_radius)/c.obstacle_margin, 0, 2)
            avoid += strength[:, None]*rel/np.maximum(d[:, None], 1e-12)
        gate = np.ones(c.n)
        if use_feedback:
            gate += c.caution_gain/(1+np.exp(-np.clip(self.s, -60, 60)))
        desired = c.alignment_gain*align + c.cohesion_gain*cohesion
        desired += gate[:, None]*(c.separation_gain*separation + c.obstacle_gain*avoid)
        desired[:, 1] += c.migration_gain
        desired[:, 0] -= c.center_gain*pos[:, 0]
        target = np.arctan2(desired[:, 1], desired[:, 0])
        target = np.where(np.linalg.norm(desired, axis=1) > 1e-12, target, theta)
        # Diffusive noise convention explicit; same seed/step/agent draws in every mode.
        target += c.noise*np.sqrt(c.dt)*noise
        if self.mode in ("full", "scalar", "passive", "scalar_matched", "instant_matched"):
            mean_z = (adj*z).sum(axis=1)/np.maximum(degree, 1)
            target_s = c.slow_base + np.tanh(c.feedback_gain*mean_z)
            self.s = target_s+(self.s-target_s)*np.exp(-c.slow_rate*c.dt)
        return target

    def step(self, pos, theta, obstacles, noise):
        c = self.c
        target = self.heading_command(pos, theta, obstacles, noise)
        turn = (target-theta+np.pi) % (2*np.pi)-np.pi
        theta = theta + np.clip(turn, -c.max_turn_rate*c.dt, c.max_turn_rate*c.dt)
        length = safe_step_lengths(pos, obstacles, c) if c.safety_braking else np.full(c.n,c.dt*c.speed)
        pos = pos + length[:, None]*np.column_stack([np.cos(theta), np.sin(theta)])
        return pos, theta
