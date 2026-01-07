# CPU Scheduling Algorithms Simulation

## Overview
This project implements and compares three CPU scheduling algorithms:
- **FCFS (First Come First Served)** - Non-preemptive, processes executed in arrival order
- **Priority Non-Preemptive** - Processes executed based on priority with aging mechanism
- **Round Robin** - Preemptive with time quantum of 150ms

## Features
- Generates 100 processes using Poisson distribution (λ=150)
- Calculates key performance metrics (waiting time, turnaround time, response time, CPU utilization, throughput)
- Produces three comprehensive visualization graphs
- Includes aging mechanism for priority scheduling to prevent starvation

## Installation
```bash
pip install matplotlib numpy tabulate
```

## Usage

### Method 1: Complete Simulation (Recommended)
```bash
python cpu_scheduling_simulation.py
```

### Method 2: Generate Missing Graphs Only
```bash
python generate_missing_graphs.py
```

## Generated Outputs

### 1. Performance Comparison (performance_comparison.png)
- Bar charts comparing average waiting time, turnaround time, and response time
- System efficiency metrics (throughput and CPU utilization)
- Numerical values displayed on each bar for precise comparison

### 2. Gantt Charts (gantt_charts.png)
- Visual timeline showing execution order for first 5 processes
- Separate charts for each scheduling algorithm
- Time labels showing burst duration for each process

### 3. Radar Chart (radar_chart.png)
- Multi-dimensional comparison across 6 performance criteria:
  - Throughput, Response Time, Fairness
  - CPU Utilization, Simplicity, Starvation Prevention
- Scores range from 1 (worst) to 10 (best)

## Sample Results
```
+-------------+---------------+------------------+----------------+------------+--------------+
| Algorithm   | Avg Waiting   | Avg Turnaround   | Avg Response   | CPU Util   |   Throughput |
+=============+===============+==================+================+============+==============+
| FCFS        | 225.9 ms      | 379.2 ms         | 225.9 ms       | 85.5%      |         5.58 |
| Priority    | 221.9 ms      | 375.1 ms         | 221.9 ms       | 85.5%      |         5.58 |
| Round Robin | 343.4 ms      | 496.7 ms         | 198.4 ms       | 85.5%      |         5.58 |
+-------------+---------------+------------------+----------------+------------+--------------+
```

## Key Findings
- **FCFS**: Best for simplicity and CPU utilization, poor response time
- **Priority**: Balanced performance with starvation prevention via aging
- **Round Robin**: Best response time and fairness, higher overhead

## Files
- `cpu_scheduling_simulation.py` - Main simulation script
- `generate_missing_graphs.py` - Standalone graph generator
- `performance_comparison.png` - Performance metrics visualization
- `gantt_charts.png` - Process execution timelines
- `radar_chart.png` - Multi-metric algorithm comparison