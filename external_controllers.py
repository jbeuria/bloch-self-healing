"""Literature-based 2D controllers and a shared bounded-velocity actuator.

These are task adaptations, not reproductions of the papers' experiments.
See EXTERNAL_BASELINES.txt for equations, sources and every major adaptation.
"""
from dataclasses import replace
import numpy as np
from .controller import Controller, geometry
from .safety import safe_step_lengths

EXTERNAL_MODES = ('full', 'olfati_saber', 'couzin', 'vasarhelyi')
INTERNAL_MODES = ('full','fast','frozen','scalar','scalar_matched','instant_matched')
PLANT = {'max_acceleration': 2.0, 'velocity_relaxation': 0.25}
DEFAULTS = {
    'full': {},
    'olfati_saber': {'spacing': 3.2/1.2, 'potential': 1., 'consensus': 1., 'repulsion_ratio': 1.,
                     'obstacle_potential': 8., 'obstacle_damping': 2., 'clearance': 2.},
    'couzin': {'repulsion_radius': .9, 'orientation_radius': 2.,
               'social_gain': 1., 'obstacle_gain': 8.},
    'vasarhelyi': {'repulsion_radius': 1.3, 'repulsion_gain': 2.,
                   'friction_offset': 2., 'friction_gain': .5, 'velocity_slack': .1,
                   'braking_acceleration': 2., 'braking_gain': 1.,
                   'shill_offset': .5, 'shill_speed': 2.,
                   'shill_acceleration': 2., 'shill_gain': 1.},
}


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def cap(v, limit):
    return v * np.minimum(1., limit / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12))


def sigma_norm(r, epsilon=.1):
    return (np.sqrt(1+epsilon*np.asarray(r)**2)-1)/epsilon


def bump(z, h=.2):
    z = np.asarray(z)
    return np.where(z < h, 1., np.where(z < 1., .5*(1+np.cos(np.pi*(z-h)/(1-h))), 0.))


def braking_curve(r, acceleration, gain):
    """Vasarhelyi Eq. 4, continuous with continuous derivative at crossover."""
    r = np.maximum(np.asarray(r), 0.)
    return np.where(r*gain < acceleration/gain, r*gain,
                    np.sqrt(np.maximum(2*acceleration*r-acceleration**2/gain**2, 0.)))


def relax_difference(dv, allowed):
    size = np.linalg.norm(dv, axis=-1)
    return dv * (np.maximum(size-allowed, 0.)/np.maximum(size, 1e-12))[..., None]


def directional_guard(pos, velocity, obstacles, c):
    """Local supporting-halfspace guard; permits outward motion at contact.

    Each agent consumes at most .45 of a pair's surface gap along the line
    of centers. The two agents together leave >= .1 of that gap. Tangential
    and outward displacement cannot violate this separating halfspace.
    Disk obstacles use .9 of their supporting-plane gap. No remote states
    or other agents' intended controls are used.
    """
    delta, distance, adj = geometry(pos, c.sensing_radius)
    normal = unit(delta)
    inward = -np.sum(normal*velocity[:, None, :], axis=-1)*c.dt
    gap = np.maximum(distance-2*c.agent_radius, 0.)
    limit = np.where(adj & (inward > 0), .45*gap/np.maximum(inward, 1e-15), np.inf)
    scale = np.minimum(1., limit.min(axis=1))
    for normal, gap, _, _ in obstacle_geometry(pos, obstacles, c):
        inward = -np.sum(normal*velocity, axis=1)*c.dt
        scale = np.minimum(scale,np.where(inward>0,.9*gap/np.maximum(inward,1e-15),np.inf))
    return velocity*scale[:,None], scale < 1-1e-12


def couzin_social(pos, heading, radius, repulsion_radius, orientation_radius):
    delta, dist, adj = geometry(pos, radius)
    rep = adj & (dist < repulsion_radius)
    orient = adj & (dist >= repulsion_radius) & (dist < orientation_radius)
    attract = adj & (dist >= orientation_radius)
    repel = (rep[..., None]*unit(delta)).sum(axis=1)
    alignment = orient @ heading
    attraction = -(attract[..., None]*unit(delta)).sum(axis=1)
    both = orient.any(axis=1) & attract.any(axis=1)
    social = (alignment+attraction)*np.where(both, .5, 1.)[:, None]
    social = np.where(rep.any(axis=1)[:, None], repel, social)
    social = np.where((np.linalg.norm(social, axis=1) < 1e-12)[:, None], heading, social)
    return unit(social)


def obstacle_geometry(pos, obstacles, c):
    for ox, oy, radius in obstacles:
        rel = pos-[ox, oy]
        distance = np.linalg.norm(rel, axis=1)
        normal = unit(rel)
        inflated = radius+c.agent_radius
        gap = np.maximum(distance-inflated, 0.)
        yield normal, gap, distance, inflated


def olfati_acceleration(pos, velocity, obstacles, c, p):
    delta, dist, adj = geometry(pos, c.sensing_radius)
    z = sigma_norm(dist)
    weights = adj*bump(z/sigma_norm(c.sensing_radius))
    shift = z-sigma_norm(p['spacing'])
    # Paper's uneven sigmoid; ratio=1 reproduces the original a=b=5 case.
    a,b=5.,5.*p.get('repulsion_ratio',1.)
    offset=(b-a)/np.sqrt(4*a*b)
    phi=weights*.5*((a+b)*(shift+offset)/np.sqrt(1+(shift+offset)**2)+(a-b))
    nij = -delta/np.sqrt(1+.1*dist**2)[..., None]
    dv = velocity[None, :, :]-velocity[:, None, :]
    acceleration = p['potential']*(phi[..., None]*nij).sum(axis=1)
    acceleration += p['consensus']*(weights[..., None]*dv).sum(axis=1)
    for normal, gap, distance, inflated in obstacle_geometry(pos, obstacles, c):
        z = sigma_norm(gap)
        d_beta = sigma_norm(p['clearance'])
        b = bump(z/d_beta, .9)*(gap < c.obstacle_margin)
        phi = b*((z-d_beta)/np.sqrt(1+(z-d_beta)**2)-1)
        n_beta = -normal*gap[:, None]/np.sqrt(1+.1*gap**2)[:, None]
        tangent = velocity-normal*np.sum(normal*velocity, axis=1)[:, None]
        v_beta = inflated/np.maximum(distance, 1e-12)[:, None] * tangent
        acceleration += p['obstacle_potential']*phi[:, None]*n_beta
        acceleration += p['obstacle_damping']*b[:, None]*(v_beta-velocity)
    # Task gamma adaptation: common centerline, forward-velocity objective.
    acceleration[:, 0] -= c.center_gain*pos[:, 0]
    acceleration += c.migration_gain*(np.array([0., c.speed])-velocity)
    return acceleration


def vasarhelyi_velocity(pos, velocity, heading, obstacles, c, p):
    delta, dist, adj = geometry(pos, c.sensing_radius)
    rep = adj*np.maximum(p['repulsion_radius']-dist, 0.)*p['repulsion_gain']
    desired = c.speed*heading+(rep[..., None]*unit(delta)).sum(axis=1)
    allowed = np.maximum(p['velocity_slack'], braking_curve(
        dist-p['friction_offset'], p['braking_acceleration'], p['braking_gain']))
    # Toward neighbors' velocities (damping), not away; see source/sign note.
    dv = velocity[None, :, :]-velocity[:, None, :]
    desired += p['friction_gain']*(adj[..., None]*relax_difference(dv, allowed)).sum(axis=1)
    for normal, gap, _, _ in obstacle_geometry(pos, obstacles, c):
        allowed = braking_curve(gap-p['shill_offset'], p['shill_acceleration'], p['shill_gain'])
        correction = relax_difference(p['shill_speed']*normal-velocity, allowed)
        desired += (gap < c.obstacle_margin)[:, None]*correction
    desired[:, 0] -= c.center_gain*pos[:, 0]
    desired[:, 1] += c.migration_gain
    return desired


class MatchedController:
    clearance_tolerance = 1e-10
    def __init__(self, config, mode, parameters=None, plant=None):
        if mode not in EXTERNAL_MODES and mode not in INTERNAL_MODES:
            raise ValueError(f'unknown external benchmark mode: {mode}')
        self.c, self.mode = config, mode
        self.p = dict(DEFAULTS['full' if mode in INTERNAL_MODES else mode])
        self.p.update(parameters or {})
        self.plant = dict(PLANT); self.plant.update(plant or {})
        if any(not np.isfinite(v) or v <= 0 for v in self.plant.values()):
            raise ValueError('plant parameters must be finite and positive')
        if mode not in INTERNAL_MODES:
            if set(self.p) != set(DEFAULTS[mode]) or any(not np.isfinite(v) or v <= 0 for v in self.p.values()):
                raise ValueError('invalid controller parameters')
        if mode == 'couzin' and not 2*config.agent_radius < self.p['repulsion_radius'] < self.p['orientation_radius'] < config.sensing_radius:
            raise ValueError('Couzin zones must be ordered inside sensing radius')
        self.inner = Controller(replace(config, noise=0.), mode) if mode in INTERNAL_MODES else None
        self.m = np.zeros((config.n,config.n,3))
        self.s = self.inner.s.copy() if self.inner is not None else np.zeros(config.n)
        self.velocity = None
        self.diagnostics = {'guard_fraction': 0., 'command_acceleration_mean': 0.,
                            'actual_acceleration_max': 0.}

    def restore_internal_state(self, m, s):
        if self.inner is not None:
            self.inner.m=np.array(m,copy=True)
            self.inner.s=np.array(s,copy=True)

    def step(self, pos, theta, obstacles, noise):
        c = self.c
        heading = np.column_stack([np.cos(theta), np.sin(theta)])
        if self.velocity is None:
            self.velocity = c.speed*heading
        old = self.velocity.copy()
        if self.mode in INTERNAL_MODES:
            angle = self.inner.heading_command(pos, theta, obstacles, np.zeros(c.n))
            desired = c.speed*np.column_stack([np.cos(angle), np.sin(angle)])
            self.m, self.s = self.inner.m, self.inner.s
        elif self.mode == 'couzin':
            desired = self.p['social_gain']*couzin_social(pos, heading, c.sensing_radius,
                        self.p['repulsion_radius'], self.p['orientation_radius'])
            desired[:,0] -= c.center_gain*pos[:,0]
            desired[:,1] += c.migration_gain
            for normal, gap, _, _ in obstacle_geometry(pos, obstacles, c):
                desired += self.p['obstacle_gain']*np.clip(1-gap/c.obstacle_margin,0,1)[:,None]*normal
            desired = c.speed*unit(desired)
        elif self.mode == 'olfati_saber':
            desired = old+c.dt*olfati_acceleration(pos,old,obstacles,c,self.p)
        else:
            desired = vasarhelyi_velocity(pos,old,heading,obstacles,c,self.p)
        if self.mode != 'olfati_saber':
            desired = old+(cap(desired,c.speed)-old)*(1-np.exp(-c.dt/self.plant['velocity_relaxation']))
        desired = cap(desired, c.speed)
        magnitude = np.linalg.norm(desired, axis=1)
        angle = np.where(magnitude > 1e-12, np.arctan2(desired[:,1],desired[:,0]), theta)
        angle += c.noise*np.sqrt(c.dt)*noise
        turn = (angle-theta+np.pi)%(2*np.pi)-np.pi
        angle = theta+np.clip(turn,-c.max_turn_rate*c.dt,c.max_turn_rate*c.dt)
        requested = magnitude[:,None]*np.column_stack([np.cos(angle),np.sin(angle)])
        velocity = old+cap(requested-old,self.plant['max_acceleration']*c.dt)
        actual, guarded = (directional_guard(pos,velocity,obstacles,c) if c.safety_braking
                           else (velocity,np.zeros(c.n,dtype=bool)))
        self.diagnostics = {'guard_fraction': float(guarded.mean()),
            'command_acceleration_mean': float(np.linalg.norm(velocity-old,axis=1).mean()/c.dt),
            'actual_acceleration_max': float(np.linalg.norm(actual-old,axis=1).max()/c.dt)}
        self.velocity = actual
        theta = np.where(np.linalg.norm(velocity,axis=1)>1e-12,np.arctan2(velocity[:,1],velocity[:,0]),theta)
        return pos+c.dt*actual, theta
