# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Waste Management System Environment.

The smart_waste_management_system environment is a simple test environment that echoes back messages.
"""

'''
For Time, we are just using an int, where it indicates the hour of day.
For Traffic, we are just using a 2D grid
'''

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, BaseModel
from typing import Tuple, List

class Truck(BaseModel):
    position: Tuple[float, float]
    max_capacity: float
    remaining_capacity: float
    #current_load is available just like other attributes (obj.current_load)
    speed: float
    fuel_remaining: float

    @property
    def current_load(self) -> float:
        return self.max_capacity - self.remaining_capacity

class Bin(BaseModel):
    position: Tuple[float, float]
    fill_level: float
    capacity: float
    fill_rate: float
    last_collected: int
    overflowed: bool

class ExternalDynamicFactors(BaseModel):
    festival: bool
    rain: bool
    peak_hours: bool
    # no_factors: bool (This is redundant since all false implies no factors.)

class SmartWasteManagementSystemState(State):
    truck: Truck
    bins: List[Bin]
    current_time: int
    traffic_grid: list[list[float]]
    external_factors: ExternalDynamicFactors

class SmartWasteManagementSystemAction(Action):
    target_bin_index: int = Field(
        ..., description="Index of the bin to visit next"
    )

class SmartWasteManagementSystemObservation(Observation):
    # Truck
    truck_position: Tuple[float, float]
    remaining_capacity: float

    # Bins
    bin_positions: List[Tuple[float, float]]
    bin_fill_levels: List[float]
    bin_fill_rates: List[float]
    time_since_last_collect: List[int]

    # Global
    time_of_day: int
    traffic_level: float

    # External (partial exposure)
    # Here, although an ExternalDynamicFactors would capture more information, 
    # its better to only expose that dataclass partially and let the agent learn the
    # extra facts, like that about rain, or festival.
    peak_hours: bool