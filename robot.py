"""
Robot data and robot behavior charts.
"""

import numpy as np
from statemachine import State, StateChart
from dataclasses import asdict

from config import Config

EPSILON = 1e-9


class Robot(Config):
    def __init__(
            self,
            robot_id: int,
            role: str,
            position: np.ndarray,
            soc: float,
            depot_position: np.ndarray,
            config: Config,
            station_pos: np.ndarray | None = None,
    ):
        super().__init__(**asdict(config))

        self.id = robot_id
        self.role = role
        self.position = position
        self.soc = soc
        self.depot_position = depot_position
        self.config = config
        self.station_pos = station_pos

        self.heading = 0.0
        self.target = self.position.copy()
        self.assigned_refueler = None
        self.robots: list[Robot]
        self.state: str | None = None  # Managed by python-statemachine.

    def update_robots(self, robots: list["Robot"]) -> None:
        self.robots = robots


class _RobotChart(StateChart[Robot]):

    def is_battery_full(self) -> bool:
        return self.model.soc >= self.model.max_battery

    def is_battery_low(self) -> bool:
        # TODO add dynamic threshold (enough to reach closest energy source - refueler/depot)
        return self.model.soc < self.model.low_soc_threshold

    def is_charging_complete(self):

        # TODO check if donor still there

        dist_home = np.linalg.norm(self.model.position - self.model.depot_position)
        min_return_energy = self.model.alpha*dist_home + self.model.safety_padding
        home_need = min_return_energy - self.model.soc  # deficit/extra energy needed to reach home

        donor = self.model.assigned_refueler
        refueler_dist_home = np.linalg.norm(donor.position - self.model.depot_position)
        refueler_min_return_energy = donor.alpha*refueler_dist_home + donor.safety_padding
        can_give = donor.soc - refueler_min_return_energy  # donor surplus

        donation = min(can_give, home_need)
        return bool(donation <= EPSILON)

    def is_battery_critical(self) -> bool:
        return self.model.soc <= EPSILON

    def has_reached_depot(self) -> bool:
        dist_home = np.linalg.norm(self.model.position - self.model.depot_position)
        return bool(dist_home <= self.model.depot_radius)

    def has_reached_station(self) -> bool:
        dist_to_station = np.linalg.norm(self.model.position - self.model.station_pos)
        return bool(dist_to_station <= self.model.station_arrival_radius)

    def has_reached_refueler(self) -> bool:
        dist_to_refueler = np.linalg.norm(self.model.position - self.model.assigned_refueler.position)
        return bool(dist_to_refueler <= self.model.refueler_radius)

    def is_v2v_enabled(self) -> bool:
        return self.model.config.v2v_enabled  # TODO read from sim

    def cannot_reach_depot(self) -> bool:
        dist_home = np.linalg.norm(self.model.position - self.model.depot_position)
        min_return_energy = self.model.alpha * dist_home + self.model.safety_padding
        return self.model.soc <= min_return_energy

    def can_revive(self) -> bool:
        # TODO implement revive, need to update refueler states too

        #     for d in np.where(state.mode == 3)[0]:
        #         need = config.alpha*np.linalg.norm(state.pos[d] - state.depot) + 0.05
        #         for j in range(N):
        #             if j == d or state.mode[j] == 3:
        #                 continue
        #             if np.linalg.norm(state.pos[j] - state.pos[d]) > config.revive_radius:
        #                 continue
        #             give = (state.soc[j]
        #                     - config.alpha*np.linalg.norm(state.pos[j] - state.depot)
        #                     - 0.02)
        #             if give < need:
        #                 continue

        return False

    def refueler_available(self) -> bool:
        """Check whether a refueler has enough energy to give and be able to return home.

        If true, set refueler id for the one we are going to travel to.
        """
        candidates = [
            robot for robot in self.model.robots
            if robot.role == "REFUELER"
               and robot.state == "available_at_station"
        ]  # Refuel robots that are available at station

        if candidates:
            # Energy robot needs to reach each refueler
            positions = np.array([donor.position for donor in candidates])
            travel_distances = np.linalg.norm(positions - self.model.position, axis=1)
            travel_energies = self.model.alpha*travel_distances + self.model.safety_padding

            # Energy each refueler can donate while preserving its return reserve.
            surplus = np.array([
                donor.soc - donor.alpha*np.linalg.norm(
                    donor.position - self.model.depot_position) - donor.safety_padding
                for donor in candidates
            ])

            reachable = travel_energies <= self.model.soc  # Donors that the robot can reach with its current energy
            eligible = reachable & (surplus > 0)           # Eligible donors that can donate surplus energy and are reachable
            if eligible.any():
                eligible_indices = np.flatnonzero(eligible)
                best_index = eligible_indices[np.argmax(surplus[eligible_indices])]
                self.model.assigned_refueler = candidates[best_index]  # highest surplus station among eligible donors
                return True

        return False

    def refueler_needs_return(self) -> bool:
        dist_home = np.linalg.norm(self.model.position - self.model.depot_position)
        min_return_energy = self.model.alpha * dist_home + self.model.safety_padding
        return self.model.soc <= min_return_energy

    def has_refueler_departed(self) -> bool:
        return self.model.assigned_refueler.state != "available_at_station"


    # Callbacks when entering a state ------------------------------------------
    def on_enter_heading_to_refueler(self) -> None:
        self.model.target = self.model.assigned_refueler.position.copy()

    def on_enter_heading_to_depot(self) -> None:
        self.model.target = self.model.depot_position.copy()

    def on_enter_heading_to_station(self) -> None:
        if self.model.station_pos is None:
            raise ValueError("A refueler requires station_pos before deployment")
        self.model.target = self.model.station_pos.copy()

    def on_enter_charging_at_refueler(self) -> None:
        self.model.target = self.model.position.copy()


class ExplorerChart(_RobotChart):
    role = "EXPLORER"

    exploring = State(initial=True)
    heading_to_depot = State()
    heading_to_refueler = State()
    charging_at_depot = State()
    charging_at_refueler = State()
    immobilized = State()

    recharged = charging_at_depot.to(exploring, cond="is_battery_full")
    charged_enough = charging_at_refueler.to(exploring, cond="is_charging_complete")
    depot_reached = heading_to_depot.to(charging_at_depot, cond="has_reached_depot")
    refueler_reached = heading_to_refueler.to(charging_at_refueler, cond="has_reached_refueler")
    depleted = immobilized.from_(heading_to_depot, heading_to_refueler, cond="is_battery_critical")
    revived = immobilized.to(heading_to_depot, cond=["is_v2v_enabled", "can_revive"])
    charge_needed = (exploring.to(heading_to_refueler, cond=["is_battery_low", "is_v2v_enabled", "cannot_reach_depot", "refueler_available"])
                     | exploring.to(heading_to_depot, cond="is_battery_low"))
    refueler_departed = charging_at_refueler.to(exploring, cond="has_refueler_departed")


class RefuelerChart(_RobotChart):
    role = "REFUELER"

    charging_at_depot = State(initial=True)
    available_at_station = State()
    heading_to_station = State()
    heading_to_depot = State()
    immobilized = State(final=True)  # No rescue for immobilized refuelers. TODO implement revive

    recharged = charging_at_depot.to(heading_to_station, cond="is_battery_full")
    depot_reached = heading_to_depot.to(charging_at_depot, cond="has_reached_depot")
    station_reached = heading_to_station.to(available_at_station, cond="has_reached_station")
    charge_needed = available_at_station.to(heading_to_depot, cond="refueler_needs_return")
    depleted = heading_to_depot.to(immobilized, cond="is_battery_critical")

    # TODO Maybe add: heading_to_station → immobilized
    # TODO Maybe add: heading_to_station → heading_to_depot (Deployment may need to be aborted)


def create_machines(robots: list[Robot]):
    """Bind charts directly to robots, which already contain their configuration."""
    return [
        (RefuelerChart if robot.role == "REFUELER" else ExplorerChart)(model=robot)
        for robot in robots
    ]
