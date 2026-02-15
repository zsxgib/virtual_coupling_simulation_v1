# Virtual Coupling Simulation

Event-triggered control simulation for train platoon with adaptive threshold schemes.

## Overview

This simulation implements a virtual coupling control system for a train platoon, comparing different event-triggered communication strategies.

## Schemes

| Scheme | Type | Description |
|--------|------|-------------|
| A | Periodic | Fixed time interval communication |
| B | Fixed Threshold | σ = constant |
| C | Error-Driven | Adaptive σ based on estimation error |
| D | State-Driven | Adaptive σ based on relative state error |
| E | Lyapunov-Driven | Adaptive σ based on Lyapunov stability |

## Running

```bash
python main.py --mode compare
```

## Results

Results are saved in `results/` directory with timestamp.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- scipy
