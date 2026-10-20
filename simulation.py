"""Pure simulation API: arrays and summaries in, no plots or filesystem writes."""
from dataclasses import dataclass
import time
import numpy as np
from .controller import Controller
from .metrics import observe, swept_clearances, detect_events, summarize
from .scenarios import make_scenario


@dataclass
class Result:
    config: dict
    summary: dict
    events: list
    series: dict
    positions: np.ndarray
    headings: np.ndarray
    snapshot_time: np.ndarray
    groups: np.ndarray
    obstacles: np.ndarray


def simulate(config, scenario="split", seed=1, mode="full", record=True, *, initial_state=None,
             controller_factory=None):
    config.validate()
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    started = time.perf_counter()
    pos, theta, groups, obstacles = make_scenario(config, scenario, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    controller = (controller_factory or Controller)(config, mode)
    if initial_state is not None:
        # Controlled interventions may supply a shared physical checkpoint and
        # controller-specific states obtained from the same observation history.
        pos = np.array(initial_state["positions"], dtype=float, copy=True)
        theta = np.array(initial_state["headings"], dtype=float, copy=True)
        groups = np.array(initial_state["groups"], dtype=int, copy=True)
        obstacles = np.array(initial_state["obstacles"], dtype=float, copy=True)
        controller.m = np.array(initial_state["m"], dtype=float, copy=True)
        controller.s = np.array(initial_state["s"], dtype=float, copy=True)
        shapes = ((pos, (config.n,2)), (theta,(config.n,)), (groups,(config.n,)),
                  (controller.m,(config.n,config.n,3)), (controller.s,(config.n,)))
        if any(x.shape != shape or not np.isfinite(x).all() for x,shape in shapes):
            raise ValueError("invalid checkpoint shape or nonfinite state")
        if obstacles.ndim != 2 or obstacles.shape[1] != 3 or not np.isfinite(obstacles).all():
            raise ValueError("invalid checkpoint obstacles")
        if np.max(np.linalg.norm(controller.m,axis=-1)) > 1+1e-10:
            raise ValueError("checkpoint outside Bloch ball")
        if np.max(abs(controller.s-config.slow_base)) > 1+1e-10:
            raise ValueError("checkpoint outside bounded self interval")
        if len(np.unique(groups)) != 2:
            raise ValueError("checkpoint requires two cohort labels")
        if hasattr(controller, 'restore_internal_state'):
            controller.restore_internal_state(controller.m, controller.s)
    rows, positions, headings, snapshot_time = [], [], [], []
    minimum_pair, minimum_obstacle = swept_clearances(pos, pos, obstacles, config.agent_radius)
    if minimum_pair < 0 or minimum_obstacle < 0:
        raise ValueError("scenario starts with overlapping agents or obstacles")
    steps = round(config.duration/config.dt)
    speeds = np.full(config.n,config.speed)
    for step in range(steps+1):
        row = observe(pos, theta, groups, obstacles, config)
        row.update(time=step*config.dt, tone=float(controller.s.mean()),
                   max_bloch_norm=float(np.linalg.norm(controller.m, axis=-1).max()),
                   mean_speed=float(speeds.mean()),
                   braked_fraction=float(np.mean(speeds < .999*config.speed)))
        if hasattr(controller, 'diagnostics'):
            row.update(controller.diagnostics)
        rows.append(row)
        if record and (step % config.record_every == 0 or step == steps):
            positions.append(pos.copy()); headings.append(theta.copy()); snapshot_time.append(step*config.dt)
        if step == steps:
            break
        after, theta = controller.step(pos, theta, obstacles, rng.normal(size=config.n))
        pc, oc = swept_clearances(pos, after, obstacles, config.agent_radius)
        minimum_pair, minimum_obstacle = min(minimum_pair, pc), min(minimum_obstacle, oc)
        speeds = np.linalg.norm(after-pos,axis=1)/config.dt
        pos = after
    series = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    events = detect_events(series["time"], series["cluster"], series["polar"], config)
    cleared = not len(obstacles) or bool(pos[:, 1].min() > np.max(obstacles[:, 1]+obstacles[:, 2])+config.agent_radius)
    summary = summarize(series, events, config, scenario, seed, mode, minimum_pair, minimum_obstacle, cleared,
                        clearance_tolerance=getattr(controller, 'clearance_tolerance', 0.))
    summary["runtime_seconds"] = time.perf_counter()-started
    return Result(config.to_dict(), summary, events, series, np.asarray(positions),
                  np.asarray(headings), np.asarray(snapshot_time), groups, obstacles)
