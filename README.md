---
title: Smart Waste Management System Environment Server
emoji: 🎙️
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Smart Waste Management System (OpenEnv Environment)

## Overview & Motivation

Urban waste management is a critical challenge in modern cities. Indian Cities are notoriously famous for their inefficient waste management. Inefficient collection routes, unpredictable waste generation, and dynamic external conditions (traffic, weather, peak hours) often lead to:

- Overflowing bins
- Increased operational costs
- Environmental and public health risks

This project models a **Smart Waste Management System** as a **reinforcement learning environment** using OpenEnv. The goal is to enable agents to learn **optimal waste collection strategies** under realistic, dynamic conditions.

This environment is designed to be:
- **Trainable** → suitable for RL agents
- **Dynamic** → influenced by real-world factors
- **Evaluatable** → includes deterministic scoring (graders)

---

## Environment Description

The environment simulates:

- A **waste collection truck**
- Multiple **waste bins** distributed in a 2D grid
- A **traffic-aware road network**
- **Dynamic external factors** (rain, festivals, peak hours)

The agent must decide **which bin to visit next** at each step.

---

## Action Space

### `SmartWasteManagementSystemAction`

```python
target_bin_index: int
```

* Represents the index of the bin the truck will visit next.
* Discrete action space: `[0, num_bins - 1]`

## Observation Space

**`SmartWasteManagementSystemObservation`**

### Truck Information
- `truck_position`: `Tuple[float, float]` — Current position of the truck
- `remaining_capacity`: `float` — Remaining waste capacity (max = 5.0)

### Bin Information
- `bin_positions`: `List[Tuple[float, float]]` — Positions of all bins
- `bin_fill_levels`: `List[float]` — Current fill level of each bin (0.0 to 1.0)
- `bin_fill_rates`: `List[float]` — Fill rate per hour for each bin
- `time_since_last_collect`: `List[int]` — Hours since last collection for each bin

### Global State
- `time_of_day`: `int` — Current hour of the day (0–23)
- `traffic_level`: `float` — Average traffic multiplier across the grid

### External Signals
- `peak_hours`: `bool` — Whether it is currently peak hours

### Metadata
- `task_type`: `str` → `"easy" | "medium" | "hard"`
- `score`: `float` — Final episode score (available at episode end)
- `reward`: `float` — Reward for the current step
- `done`: `bool` — Whether the episode has ended

---

## Environment Dynamics

- **Bins** fill continuously based on their `fill_rate`, increased during festivals and peak hours.
- **Traffic** is modeled as a 2D grid of multipliers (0.8 – 1.2+). Higher values = slower movement.
- **Travel time** is calculated using **A*** pathfinding on the traffic grid.
- **Time advances** based on the computed travel cost of the chosen route.
- **Overflow** occurs when a bin's `fill_level` reaches or exceeds its `capacity` (1.0).

---

## Tasks & Difficulty Levels for the Programmatic Grader 

Each episode randomly assigns one of the following difficulty levels:

### Easy Task
**Objective**: Prevent all bin overflows  
**Scoring**:
```math
\text{Score} = 1.0 \text{ if no overflows, else } 0.0
```

### Medium Task
**Objective**: Prevent overflows + Minimize travel cost

**Scoring**:
```math
Score = 0.6 \times (1 - overflow\_ratio) + 0.4 \times travel\_score 
```

### Hard Task
**Objective**: Prevent overflows + Minimize travel cost + Maximize collection efficiency

**Scoring**:
```math
Score = 0.4 \times (1 - overflow\_ratio) + 0.3 \times travel\_score + 0.3 \times collection\_efficiency
```

## Reward Function

Current Reward Function (per step):

```math
reward = 
30.0 \times collected 
- 0.6 \times path\_cost 
- 0.02 \times path\_length 
- 120.0 \times overflow\_penalty 
- 0.5
```

## Setup Instructions