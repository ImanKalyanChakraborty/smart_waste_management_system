# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smart Waste Management System Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import random
import math
from typing import Optional, Any, List, Tuple, Dict
import heapq

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartWasteManagementSystemAction, SmartWasteManagementSystemObservation, SmartWasteManagementSystemState, Truck, Bin, ExternalDynamicFactors
except ImportError:
    from models import SmartWasteManagementSystemAction, SmartWasteManagementSystemObservation, SmartWasteManagementSystemState, Truck, Bin, ExternalDynamicFactors


class SmartWasteManagementSystemEnvironment(Environment):
    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, num_bins: int = 5, grid_size: int = 10):
        """Initialize the smart_waste_management_system environment."""
        self._state: SmartWasteManagementSystemState | None = None
        self._reset_count = 0
        self.num_bins = num_bins
        self.grid_size = grid_size
        self.max_time = 24  # one episode = 1 day

        # Programmatic Grader Fields
        self.task_type = "easy"  # default
        self.total_overflows = 0
        self.total_travel_cost = 0.0
        self.total_collections = 0

    def reset(self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any
        ) -> SmartWasteManagementSystemObservation:
        """
        Reset the environment.

        Returns:
            SmartWasteManagementSystemObservation with a ready message
        """
        self._reset_count += 1

        if seed is not None:
            random.seed(seed)

        episode_id = episode_id or str(uuid4())

        bins = []
        for _ in range(self.num_bins):
            bins.append(
                Bin(
                    position=(random.uniform(0, self.grid_size),
                              random.uniform(0, self.grid_size)),
                    fill_level=random.uniform(0, 0.5),
                    capacity=1.0,
                    fill_rate=random.uniform(0.01, 0.05),
                    last_collected=0,
                    overflowed=False
                )
            )

        truck = Truck(
            position=(0.0, 0.0),
            max_capacity=5.0,
            remaining_capacity=5.0,
            speed=1.0,
            fuel_remaining=100.0
        )

        external_factors = ExternalDynamicFactors(
            festival=False,
            rain=False,
            peak_hours=False
        )

        # This grid is not just a map of paths, it is a difficulty grid:
        # Each cell stores a traffic multiplier, meaning 'How slow is movement through this point'.
        # travel time = distance * traffic multiplier (the value in the grid)
        # < 1 value means lower time, > 1 value means high traffic, higher time
        traffic_grid = [
            [random.uniform(0.8, 1.2) for _ in range(self.grid_size)]
            for _ in range(self.grid_size)
        ]

        self._state = SmartWasteManagementSystemState(
            episode_id=episode_id, 
            step_count=0,
            truck=truck,
            bins=bins,
            current_time=0,
            traffic_grid=traffic_grid,
            external_factors=external_factors
            )

        self.task_type = random.choice(["easy", "medium", "hard"])

        self.total_overflows = 0
        self.total_travel_cost = 0.0
        self.total_collections = 0

        return self._get_observation()

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring grid cells (4-directional movement)."""
        x, y = pos
        neighbors = []
        
        # 4-directional movement (up, down, left, right)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                neighbors.append((new_x, new_y))
        
        # Left for now: Diagonal movement with cost sqrt(2)
        # for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        #     new_x, new_y = x + dx, y + dy
        #     if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
        #         neighbors.append((new_x, new_y))
        
        return neighbors

    def _get_cell_cost(self, cell: Tuple[int, int]) -> float:
        """Get the traffic cost for a specific grid cell."""
        x, y = cell
        return self.state.traffic_grid[x][y]

    def _astar_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], float]:
        """
        A* pathfinding algorithm.
        
        Returns:
            Tuple of (path as list of grid cells, total cost)
        """
        if start == goal:
            return [start], 0.0
        
        # Priority queue: (f_cost, counter, current_node)
        counter = 0
        open_set = [(0.0, counter, start)]
        
        # Track visited nodes and their costs
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, goal)}
        
        closed_set = set()
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, g_score[goal]
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost to neighbor
                # Use the cost of the neighbor cell (where you're moving INTO)
                # Alternatively, you could average start and end cell costs
                move_cost = self._get_cell_cost(neighbor)
                
                # For diagonal movement, use sqrt(2) * cost
                # if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                #     move_cost *= math.sqrt(2)
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found (shouldn't happen in a fully connected grid)
        return [], float('inf')

    def _world_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates (continuous) to grid cell indices."""
        # Assuming grid cells are unit squares from (0,0) to (grid_size, grid_size)
        x = int(position[0])
        y = int(position[1])
        
        # Clamp to grid boundaries
        x = max(0, min(x, self.grid_size - 1))
        y = max(0, min(y, self.grid_size - 1))
        
        return (x, y)

    def _grid_to_world(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid cell to world coordinates (center of cell)."""
        # Return center of cell for smoother movement
        return (cell[0] + 0.5, cell[1] + 0.5)
    
    def _calculate_travel_time_astar(self, start_pos: Tuple[float, float], goal_pos: Tuple[float, float]) -> Tuple[float, float, List[Tuple[int, int]]]:
        """
        Calculate travel time using A* on the traffic grid.
        
        Returns:
            Tuple of (travel_time, total_path_cost, path_cells)
        """
        start_cell = self._world_to_grid(start_pos)
        goal_cell = self._world_to_grid(goal_pos)
        
        # Find optimal path using A*
        path_cells, total_cost = self._astar_pathfinding(start_cell, goal_cell)
        
        if not path_cells:
            # Fallback to Euclidean distance if pathfinding fails
            distance = self._distance(start_pos, goal_pos)
            avg_traffic = self._average_traffic()
            return distance * avg_traffic, distance * avg_traffic, []
        
        # Convert path cells to world positions (optional, for visualization)
        # path_world = [self._grid_to_world(cell) for cell in path_cells]
        
        # Travel time is the sum of traffic costs along the path
        # Each step cost represents the time to traverse that cell
        travel_time = total_cost
        
        # Add the cost of the starting cell? (depends on interpretation)
        # Current implementation: cost of moving INTO a cell
        # You might want to include start cell cost if truck hasn't moved yet
        
        return travel_time, total_cost, path_cells

    def _get_observation(self) -> SmartWasteManagementSystemObservation:
        state = self.state
        truck = state.truck

        bin_positions = [b.position for b in state.bins]
        bin_fill_levels = [b.fill_level for b in state.bins]
        bin_fill_rates = [b.fill_rate for b in state.bins]
        time_since_last_collect = [
            self.state.current_time - b.last_collected
            for b in self.state.bins
        ]

        traffic_level = self._average_traffic()

        return SmartWasteManagementSystemObservation(
            truck_position=truck.position,
            remaining_capacity=truck.remaining_capacity,

            bin_positions=bin_positions,
            bin_fill_levels=bin_fill_levels,
            bin_fill_rates=bin_fill_rates,
            time_since_last_collect=time_since_last_collect,

            time_of_day=self.state.current_time % 24,
            traffic_level=traffic_level,

            peak_hours=self.state.external_factors.peak_hours,

            metadata={
                "task_type": self.task_type
            }
        )

    def step(self, action: SmartWasteManagementSystemAction, timeout_s: float | None = None, **kwargs: Any) -> SmartWasteManagementSystemObservation:
        """Execute a step using A* pathfinding."""
        self.state.step_count += 1

        state = self.state
        target_idx = action.target_bin_index
        target_bin = state.bins[target_idx]
        
        # --------- CALCULATE PATH USING A* ----------
        travel_time, path_cost, path_cells = self._calculate_travel_time_astar(
            self.state.truck.position, 
            target_bin.position
        )
        
        if not path_cells:
        # Fallback (very rare)
            distance = self._distance(state.truck.position, target_bin.position)
            travel_time = distance * self._average_traffic()
            path_cost = travel_time
            path_cells = []

        # Move truck immediately (cleaner)
        state.truck.position = target_bin.position

        # --------- ADVANCE TIME ----------
        # IMPORTANT FIX: Advance time properly
        time_advance = max(1, int(math.ceil(travel_time)))
        state.current_time += time_advance

        # Update environment dynamics during the "trip"
        self._update_bins()
        self._update_external_factors()

        # --------- COLLECT WASTE ----------
        collected = min(target_bin.fill_level, self.state.truck.remaining_capacity)
        target_bin.fill_level -= collected
        self.state.truck.remaining_capacity -= collected
        target_bin.last_collected = self.state.current_time
        target_bin.overflowed = False
        
        # Data for programmatic grader
        self.total_travel_cost += path_cost
        self.total_collections += 1
        
        # --------- FINAL ENVIRONMENT UPDATE ----------
        self._update_external_factors()
        self._update_traffic()
        overflow_penalty = self._update_bins()

        # Data for programmatic grader
        self.total_overflows = sum(1 for b in state.bins if b.overflowed)
        
        # --------- REWARD CALCULATION ----------
        reward = (
            30.0 * collected                    # Strong positive for collecting
            - 0.6 * path_cost                   # Reduced travel penalty
            - 0.02 * len(path_cells)            # Very small step penalty
            - 120.0 * overflow_penalty          # Strong overflow penalty
            - 0.5                               # Small cost per action
        )
        
        # --------- DONE ----------
        done = self.state.current_time >= self.max_time

        score = self._compute_score()
        
        return SmartWasteManagementSystemObservation(
            truck_position=self.state.truck.position,
            remaining_capacity=self.state.truck.remaining_capacity,

            bin_positions=[b.position for b in self.state.bins],
            bin_fill_levels=[b.fill_level for b in self.state.bins],
            bin_fill_rates=[b.fill_rate for b in self.state.bins],
            time_since_last_collect=[
                self.state.current_time - b.last_collected
                for b in self.state.bins
            ],

            time_of_day=self.state.current_time % 24,
            traffic_level=self._average_traffic(),

            peak_hours=self.state.external_factors.peak_hours,

            # REQUIRED
            reward=reward,
            done=done,
            metadata={
                "task_type": self.task_type,
                "score": score,
                "travel_time": travel_time,
                "collected": collected
            }
        )

    # ------------------------------------------------------
    # GRADER SCORE BASED ON "EASY", "MEDIUM" OR "HARD" TASK
    # ------------------------------------------------------

    def _compute_score(self) -> int:
        num_bins = len(self.state.bins)

        overflow_ratio = self.total_overflows / num_bins

        # Normalize travel (assume worst-case bound)
        max_travel = self.max_time * self.grid_size * 2
        travel_score = 1 - self._normalize(self.total_travel_cost, max_travel)

        collection_efficiency = self._normalize(self.total_collections, self.max_time)

        if self.task_type == "easy":
            # ONLY overflow matters
            score = 1.0 if overflow_ratio == 0 else 0.0

        elif self.task_type == "medium":
            # Prevent overflows + minimize travel cost
            # Success only if no overflows AND travel is reasonably good
            no_overflow = (self.total_overflows == 0)
            good_travel = (travel_score >= 0.7)   # Adjust threshold as needed
            score = 1.0 if (no_overflow and good_travel) else 0.0

        else:  # hard
            # Prevent overflows + good travel + decent collection efficiency
            no_overflow = (self.total_overflows == 0)
            good_travel = (travel_score >= 0.65)
            good_efficiency = (collection_efficiency >= 0.6)
            score = 1.0 if (no_overflow and good_travel and good_efficiency) else 0.0

        return int(max(0.0, min(score, 1.0)))

    # -------------------------------------------------
    # DYNAMICS
    # -------------------------------------------------
    def _update_bins(self) -> int:
        overflow_count = 0

        for b in self.state.bins:
            factor = 1.0

            if self.state.external_factors.festival:
                factor += 0.5
            if self.state.external_factors.peak_hours:
                factor += 0.3

            b.fill_level += b.fill_rate * factor + random.uniform(0, 0.02)

            if b.fill_level >= b.capacity:
                b.fill_level = b.capacity
                b.overflowed = True
                overflow_count += 1

        return overflow_count

    def _update_traffic(self):
        base = 1.0

        if self.state.external_factors.peak_hours:
            base += 0.5
        if self.state.external_factors.rain:
            base += 0.3

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.state.traffic_grid[i][j] = base + random.uniform(-0.1, 0.1)

    def _update_external_factors(self):
        t = self.state.current_time % 24

        self.state.external_factors.peak_hours = (8 <= t <= 10) or (17 <= t <= 20)
        self.state.external_factors.rain = random.random() < 0.1
        self.state.external_factors.festival = random.random() < 0.05

    # -------------------------------------------------
    # HELPERS
    # -------------------------------------------------
    def _distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _get_traffic_multiplier(self, position):
        x = int(position[0]) % self.grid_size
        y = int(position[1]) % self.grid_size
        return self.state.traffic_grid[x][y]

    def _average_traffic(self):
        total = sum(sum(row) for row in self.state.traffic_grid)
        return total / (self.grid_size * self.grid_size)

    def _normalize(self, value, max_value):
        return min(value / max_value, 1.0) if max_value > 0 else 0.0

    @property
    def state(self) -> SmartWasteManagementSystemState:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        assert self._state is not None, "Environment not initialized. Call reset() first."
        return self._state
