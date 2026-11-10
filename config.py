"""Simulation parameters."""

import math
from dataclasses import dataclass


@dataclass()
class Config:

    # Robot
    max_battery: float = 1.0
    alpha: float = 0.0024     # battery depletion rate per unit distance traveled
    low_soc_threshold: float = 0.25
    safety_padding: float = 0.02  # extra SoC to avoid running dry on the depot's pad

    # Environment
    num_robots: int = 12
    num_refuelers: int = 1
    refueler_radius: float = 0.5
    task_rate: float = 0.06
    task_energy_cost: float = 0.18
    transfer_rate: float = 0.15
    world_size: float = 500.0
    depot_radius: float = 3.0
    recharge_rate: float = 0.004
    task_engage_radius: float = 1.5
    revive_radius: float = 3.0
    comm_range: float = 10.0
    safety_reserve: float = 0.02
    station_arrival_radius: float = 0.1

    # Simulation
    v2v_enabled: bool = True
    immobilize_on_empty: bool = True
    move_step: float = 0.5
    max_steps: int = 2000
    task_ttl: int = 200
    max_tasks: int = 30


    def __post_init__(self):
        if not math.isfinite(self.max_battery) or self.max_battery <= 0:
            raise ValueError(f"max_battery must be finite and positive")
        if not 0 <= self.low_soc_threshold < self.max_battery:
            raise ValueError("Invalid low battery SoC threshold")
        if self.num_robots < 1:
            raise ValueError(f"Fleet must be nonempty. Got num_robots = {self.num_robots}.")
        if not 0 <= self.num_refuelers <= self.num_robots:
            raise ValueError(f"Number of refuelers must be between 0 and total number of robots ({self.num_robots}). Got = {self.num_refuelers}.")


    @property
    def solo_reach(self) -> float:
        return self.max_battery / (2 * self.alpha)
