"""Recovery detection, censoring, continuous-segment safety and objective."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .controller import geometry


def observe(pos, theta, groups, obstacles, config):
    _, dist, adj = geometry(pos, config.sensing_radius)
    _, labels = connected_components(csr_matrix(adj), directed=False)
    fraction = np.bincount(labels).max()/len(pos)
    polar = abs(np.exp(1j*theta).mean())
    inter = groups[:, None] != groups[None, :]
    cross_gap = float(dist[inter].min())
    return {"cluster": float(fraction), "polar": float(polar),
            "cross_gap": cross_gap, "cross_edge": bool(np.any(adj & inter)),
            "centroid_y": float(pos[:, 1].mean()),
            "mean_degree": float(adj.sum(axis=1).mean())}


def swept_clearances(before, after, obstacles, radius):
    """Exact closest clearance on straight segments between numerical positions.

    Negative values indicate physical overlap. Checks every pair, not just sensed
    neighbors. This checks the discretized path, not continuous curved flight.
    """
    a = before[:, None, :]-before[None, :, :]
    v = (after-before)[:, None, :]-(after-before)[None, :, :]
    u = np.clip(-np.sum(a*v, axis=-1)/np.maximum(np.sum(v*v, axis=-1), 1e-20), 0, 1)
    distance = np.linalg.norm(a+u[..., None]*v, axis=-1)
    np.fill_diagonal(distance, np.inf)
    pair_clearance = float(distance.min()-2*radius)
    obs_clearance = float("inf")
    move = after-before
    for ox, oy, r in obstacles:
        rel = before-[ox, oy]
        u = np.clip(-np.sum(rel*move, axis=1)/np.maximum(np.sum(move*move, axis=1), 1e-20), 0, 1)
        d = np.linalg.norm(rel+u[:, None]*move, axis=1)-r-radius
        obs_clearance = min(obs_clearance, float(d.min()))
    return pair_clearance, obs_clearance


def detect_events(time, cluster, polar, config):
    """One active event at a time, including initial splits and terminal censoring.

    An event begins below split_fraction and ends at the START of the first
    recovery interval lasting hold_time. A further split opens a new event.
    """
    events = []
    start = None
    streak = None
    for i, t in enumerate(time):
        if start is None and cluster[i] < config.split_fraction:
            start = i
        if start is None:
            continue
        good = cluster[i] >= config.recovery_fraction and polar[i] >= config.recovery_polar
        if good:
            if streak is None:
                streak = i
            if t-time[streak] >= config.hold_time-1e-9:
                events.append({"split_time": float(time[start]), "recovery_time": float(time[streak]),
                               "confirmation_time": float(t), "duration": float(time[streak]-time[start]),
                               "observed": True, "minimum_cluster": float(min(cluster[start:i+1]))})
                start = streak = None
        else:
            streak = None
    if start is not None:
        events.append({"split_time": float(time[start]), "recovery_time": None,
                       "confirmation_time": None, "duration": float(time[-1]-time[start]),
                       "observed": False, "minimum_cluster": float(min(cluster[start:]))})
    return events


def summarize(series, events, config, scenario, seed, mode, min_pair, min_obstacle, cleared,
              clearance_tolerance=0.):
    t = series["time"]
    fraction = series["cluster"]
    horizon = float(t[-1])
    # Trapezoidal integration compatible with NumPy 1.26 and 2.x.
    deficit = float(np.sum((2-fraction[:-1]-fraction[1:])*np.diff(t)/2)/horizon)
    disconnected = float(np.mean(fraction < config.recovery_fraction))
    progress = float((series["centroid_y"][-1]-series["centroid_y"][0])/(config.speed*horizon))
    recovered = sum(e["observed"] for e in events)
    first_duration = events[0]["duration"] if events else None
    first_recovered = bool(events and events[0]["observed"])
    # Censored trials contribute the whole available horizon; no survivor-only mean.
    restricted = first_duration/horizon if first_recovered else 1.0
    if not events:
        restricted = 0.0  # prevention is recorded separately; never labeled healing
    loss = 0.45*deficit + 0.35*restricted + 0.15*disconnected + 0.05*max(0, 1-progress)
    safe = min_pair >= -clearance_tolerance and min_obstacle >= -clearance_tolerance
    moving = progress >= config.min_progress_fraction
    final_window = t >= t[-1]-config.hold_time-1e-9
    final_good = bool(np.all(fraction[final_window] >= config.recovery_fraction)
                      and np.all(series["polar"][final_window] >= config.recovery_polar))
    healing = bool(events and recovered == len(events) and final_good and safe and moving and cleared)
    return {"scenario": scenario, "seed": seed, "controller": mode,
            "loss": float(loss), "split_events": len(events), "recovered_events": recovered,
            "first_recovery_time": first_duration if first_recovered else None,
            "first_event_observed": first_recovered, "first_event_followup": first_duration,
            "connectivity_deficit": deficit, "disconnected_fraction": disconnected,
            "progress_fraction": progress, "final_cluster": float(fraction[-1]),
            "mean_speed_fraction": float(np.mean(series["mean_speed"][1:])/config.speed),
            "braked_agent_step_fraction": float(np.mean(series["braked_fraction"][1:])),
            "min_cluster": float(min(fraction)), "min_pair_clearance": min_pair,
            "min_obstacle_clearance": min_obstacle if np.isfinite(min_obstacle) else None,
            "max_bloch_norm": float(max(series["max_bloch_norm"])),
            "safe": bool(safe), "progress_ok": bool(moving), "obstacles_cleared": bool(cleared),
            "healing_success": healing,
            "status": "healed" if healing else ("no_split" if not events else "not_healed")}
