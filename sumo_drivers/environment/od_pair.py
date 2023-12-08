import random as rd
from datetime import datetime
from numpy import float64

import polars as pl

from sumo_drivers.environment.vehicle import Vehicle


class ODPair:
    """Class responsible for holding information regarding OD-pairs that are necessary in the environment.

    Args:
        straight_distance (float): distance between the origin and destination as a straight line.
    """

    PARAMS = ["Step", "Avg Travel Time", "Avg Waiting Time", "Load"]
    SCHEMAS = {
        "Step": pl.Int64,
        "Avg Travel Time": pl.Float64,
        "Avg Waiting Time": pl.Float64,
        "Load": pl.Int64,
    }

    def __init__(self, straight_distance: float, collect: bool) -> None:
        self.__straight_distance: float = straight_distance
        self.__min_load: int = -1
        self.__current_load: list[str] = []
        self.__vehicles_within: list[str] = []
        self.__vehicles_step_data: pl.DataFrame | None = (
            pl.DataFrame(
                {"Travel time": [], "Waiting time": []},
                schema={"Travel time": pl.Float64, "Waiting time": pl.Float64},
            )
            if collect
            else None
        )
        self.__step_od_data: pl.DataFrame | None = (
            pl.DataFrame({param: [] for param in self.PARAMS}, schema=self.SCHEMAS)
            if collect
            else None
        )

        rd.seed(datetime.now().timestamp())

    @property
    def min_load(self) -> int:
        """Property that returns the OD-pair minimum load required (number of vehicles running between the OD-pair).

        Returns:
            int: Minimum number of vehicles required to be running within the OD-pair
        """
        return self.__min_load

    @min_load.setter
    def min_load(self, val: int) -> None:
        """Setter for the minimum load of the OD-pair.

        Args:
            val (int): value to set the minimum load on the OD-pair

        Raises:
            RuntimeError: the method raises a RuntimeError if the value is negative.
        """
        if val <= 0:
            raise RuntimeError("Value should be positive!")

        self.__min_load = val

    @property
    def curr_load(self) -> int:
        return len(self.__current_load)

    @property
    def has_enough_vehicles(self) -> bool:
        """Property that returns a boolean indicating if the OD-pair has enough vehicles running within the OD-pair.

        Returns:
            bool: value that indicates if the OD-pair's current load is higher or equal than the minimum required.
        """
        return len(self.__current_load) >= self.__min_load

    @property
    def straight_distance(self) -> float:
        """Property that returns the straight line distance between the origin and destination of the OD-pair.

        Returns:
            float: straight distance between origin and destination of the OD-pair.
        """
        return self.__straight_distance

    def update_vehicle_data(self, vehicles: list[Vehicle]):
        if self.__vehicles_step_data is None:
            return
        vehicles_data = [
            pl.DataFrame(
                {
                    "Travel time": [vehicle.step_data["Travel time"]],
                    "Waiting time": [vehicle.step_data["Waiting time"]],
                }
            )
            for vehicle in vehicles
            if vehicle.step_data is not None
        ]
        self.__vehicles_step_data = pl.concat(
            [self.__vehicles_step_data, *vehicles_data]
        )

    def summarize_step(self, current_step):
        if self.__vehicles_step_data is None or self.__step_od_data is None:
            return

        if self.__vehicles_step_data.shape[0] == 0:
            return

        vehicles_mean = self.__vehicles_step_data.mean()
        self.__step_od_data = pl.concat(
            [
                self.__step_od_data,
                pl.DataFrame(
                    {
                        "Step": [current_step],
                        "Avg Travel Time": [vehicles_mean["Travel time"][0]],
                        "Avg Waiting Time": [vehicles_mean["Waiting time"][0]],
                        "Load": [self.curr_load],
                    }
                ),
            ]
        )
        self.__vehicles_step_data = pl.DataFrame(
            {"Travel time": [], "Waiting time": []},
            schema={"Travel time": pl.Float64, "Waiting time": pl.Float64},
        )

    def increase_load(self, vehicle_id: str) -> None:
        """Method that increases the OD-pair current load by 1."""
        self.__current_load.append(vehicle_id)

    def decrease_load(self, vehicle_id) -> None:
        """Method that decreases the OD-pair current load by 1."""
        self.__current_load.remove(vehicle_id)

    def reset(self) -> None:
        """Method that resets the OD-pair (sets the current load to 0)."""
        self.__current_load = []

    def append_vehicle(self, vehicle_id: str) -> None:
        self.__vehicles_within.append(vehicle_id)

    def random_vehicle(self) -> str:
        return rd.choice(
            [
                vehicle_id
                for vehicle_id in self.__vehicles_within
                if vehicle_id not in self.__current_load
            ]
        )

    def summarize_all(self, window_size: int) -> pl.DataFrame:
        if self.__step_od_data is None:
            return pl.DataFrame()

        return self.__step_od_data[::window_size]
