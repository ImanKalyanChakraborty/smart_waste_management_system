# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Waste Management System Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    SmartWasteManagementSystemAction,
    SmartWasteManagementSystemObservation,
)


class SmartWasteManagementSystemEnv(
    EnvClient[
        SmartWasteManagementSystemAction,
        SmartWasteManagementSystemObservation,
        State
    ]
):

    def _step_payload(self, action: SmartWasteManagementSystemAction) -> Dict:
        """
        Convert action to JSON payload.
        """
        return {
            "target_bin_index": action.target_bin_index,
        }

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[SmartWasteManagementSystemObservation]:
        """
        Parse server response into StepResult.
        """
        obs_data = payload.get("observation", {})

        observation = SmartWasteManagementSystemObservation(
            truck_position=tuple(obs_data.get("truck_position", (0.0, 0.0))),
            remaining_capacity=obs_data.get("remaining_capacity", 0.0),

            bin_positions=[
                tuple(pos) for pos in obs_data.get("bin_positions", [])
            ],
            bin_fill_levels=obs_data.get("bin_fill_levels", []),
            bin_fill_rates=obs_data.get("bin_fill_rates", []),
            time_since_last_collect=obs_data.get("time_since_last_collect", []),

            time_of_day=obs_data.get("time_of_day", 0),
            traffic_level=obs_data.get("traffic_level", 1.0),

            peak_hours=obs_data.get("peak_hours", False),
            task_score=obs_data.get("task_score", 0.0),

            # OpenEnv-required fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    # def _parse_result(
    # self, payload: Dict
    # ) -> StepResult[SmartWasteManagementSystemObservation]:
    #     # 🔍 Print the raw payload from server
    #     print(f"[DEBUG] RAW PAYLOAD: {payload}", flush=True)
        
    #     obs_data = payload.get("observation", {})
    #     print(f"[DEBUG] RAW OBS_DATA: {obs_data}", flush=True)
    #     print(f"[DEBUG] METADATA IN OBS_DATA: {obs_data.get('metadata', 'NOT FOUND')}", flush=True)
        
    #     observation = SmartWasteManagementSystemObservation(
    #         truck_position=tuple(obs_data.get("truck_position", (0.0, 0.0))),
    #         remaining_capacity=obs_data.get("remaining_capacity", 0.0),
    #         bin_positions=[
    #             tuple(pos) for pos in obs_data.get("bin_positions", [])
    #         ],
    #         bin_fill_levels=obs_data.get("bin_fill_levels", []),
    #         bin_fill_rates=obs_data.get("bin_fill_rates", []),
    #         time_since_last_collect=obs_data.get("time_since_last_collect", []),
    #         time_of_day=obs_data.get("time_of_day", 0),
    #         traffic_level=obs_data.get("traffic_level", 1.0),
    #         peak_hours=obs_data.get("peak_hours", False),
    #         done=payload.get("done", False),
    #         reward=payload.get("reward"),
    #         metadata=obs_data.get("metadata", {}),
    #     )
        
    #     print(f"[DEBUG] PARSED OBS METADATA: {observation.metadata}", flush=True)
        
    #     return StepResult(
    #         observation=observation,
    #         reward=payload.get("reward"),
    #         done=payload.get("done", False),
    #     )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )