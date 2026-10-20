"""Explicit, JSON-serializable configuration in dimensionless units."""
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

MODES = ("full", "scalar", "fast", "reactive", "passive", "frozen",
         "scalar_matched", "instant_matched")
SCENARIOS = ("split", "open", "single", "double", "triple")


@dataclass(frozen=True)
class Config:
    n: int = 48
    dt: float = 0.05
    duration: float = 90.0
    speed: float = 1.0
    safety_braking: bool = True  # set False to recover fixed-speed motion
    sensing_radius: float = 3.2
    agent_radius: float = 0.18
    initial_spacing: float = 1.0
    initial_jitter: float = 0.07
    split_gap: float = 8.0  # minimum lateral gap between imposed subgroups
    split_angle: float = 0.15  # radians outward from migration direction
    noise: float = 0.025  # heading-angle diffusion, radians / sqrt(time)
    max_turn_rate: float = 2.5
    alignment_gain: float = 1.2
    cohesion_gain: float = 0.08
    separation_gain: float = 12.0
    separation_radius: float = 1.3
    migration_gain: float = 0.8
    center_gain: float = 0.035  # explicit environmental assistance; zero = no centering
    obstacle_gain: float = 8.0
    obstacle_margin: float = 3.0
    obstacle_radius: float = 4.0
    obstacle_first_offset: float = 8.0
    obstacle_row_spacing: float = 14.0
    gamma: float = 1.0
    t1: float = 0.6
    t2: float = 0.3
    kappa: float = 1.2
    g_align: float = 2.0
    slow_rate: float = 0.03
    feedback_gain: float = 2.0
    slow_base: float = 0.0
    frozen_tone: float = 0.75
    caution_gain: float = 0.0  # scales obstacle/separation ONLY, not cohesion
    channel_policy: str = "reset"  # reset, retain, decay
    recovery_fraction: float = 0.9
    split_fraction: float = 0.85
    recovery_polar: float = 0.9
    hold_time: float = 2.0
    min_progress_fraction: float = 0.5
    record_every: int = 4  # metrics remain at every step

    def validate(self):
        for f in fields(self):
            x = getattr(self, f.name)
            if isinstance(x, (int, float)):
                import math
                if not math.isfinite(x):
                    raise ValueError(f"{f.name} must be finite")
        if self.n < 8 or int(self.n) != self.n:
            raise ValueError("n must be an integer >= 8")
        if self.record_every < 1 or int(self.record_every) != self.record_every:
            raise ValueError("record_every must be a positive integer")
        for name in ("dt", "duration", "speed", "sensing_radius", "agent_radius",
                     "initial_spacing", "max_turn_rate", "separation_radius",
                     "obstacle_margin", "obstacle_radius", "obstacle_first_offset",
                     "obstacle_row_spacing", "t1", "t2", "hold_time"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("initial_jitter", "noise", "alignment_gain", "cohesion_gain",
                     "separation_gain", "migration_gain", "center_gain", "obstacle_gain",
                     "gamma", "kappa", "g_align", "slow_rate", "feedback_gain", "caution_gain"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.t2 > 2 * self.t1:
            raise ValueError("Bloch-ball preservation requires t2 <= 2*t1")
        if not 0 < self.split_fraction < self.recovery_fraction <= 1:
            raise ValueError("require 0 < split_fraction < recovery_fraction <= 1")
        if not 0 < self.recovery_polar <= 1 or not 0 <= self.min_progress_fraction <= 1:
            raise ValueError("invalid polar/progress threshold")
        if self.split_gap <= self.sensing_radius:
            raise ValueError("imposed split gap must exceed sensing radius")
        if self.initial_spacing - 2*self.initial_jitter <= 2*self.agent_radius:
            raise ValueError("initial lattice can contain overlapping agents")
        if self.duration < self.hold_time or self.dt > self.hold_time:
            raise ValueError("require dt <= hold_time <= duration")
        if self.channel_policy not in ("reset", "retain", "decay"):
            raise ValueError("unknown channel policy")
        if not isinstance(self.safety_braking, bool):
            raise ValueError("safety_braking must be true or false")
        if self.sensing_radius <= 2*self.agent_radius+2*self.speed*self.dt:
            raise ValueError("sensing radius must cover possible one-step pair contacts")
        steps = self.duration / self.dt
        if abs(steps - round(steps)) > 1e-7:
            raise ValueError("duration must be a multiple of dt")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
