# RoadSentinel - CARLA Drone Simulation Testbed v3.0

High-altitude aerial road-inspection drone simulation capturing geo-tagged, overlap-controlled aerial imagery for the RoadSentinel AI inspection pipeline (DINOv2 anomaly detection + SAM2 + 3m spatial deduplication + VLM work orders).

## New Interactive Launcher

Launch via the interactive pre-CARLA TUI menu:
```bash
python run_simulation.py
```
Or directly from the command line:
```bash
python run_simulation.py --non-interactive --scenario poor --weather post_rain --altitude 100.0 --speed 30.0 --seed 42 --duration 60.0
```

## Core Architecture Files

| File | Purpose |
|---|---|
| `run_simulation.py` | Pre-CARLA interactive TUI launcher, Docker manager, and session orchestrator |
| `env/config.py` | Standardized parameters (100m altitude, 60° FOV, calibrated weather presets, monotonic scenarios) |
| `env/road_injector.py` | 3D depression geometry engine (6 shape families), water rendering (turbidity, capillary halos), road resurfacing patches |
| `env/drone_sim.py` | Drone flight rig (live CARLA and standalone procedural mode, image overlap capture, defect projection) |
| `env/drone_controller.py` | Kinematic flight controls (configurable survey speed, yaw, corridor cycling) |
| `env/overlap_calculator.py` | 70% overlap capture interval and GSD math |
| `env/evaluate_simulation.py` | Quantitative evaluation comparing ML predictions against CARLA ground truth |
| `env/generate_road_scenarios.py` | Batch dataset generator across scenario × weather × seed × altitude axes |
| `env/geo_utils.py` | Standardized WGS-84 coordinate transforms and datum projection |

## Drone Flight Controls (Manual Flight Mode)
```
W / S       forward / backward (survey speed)
A / D       strafe left / right
Q / E       yaw left / right
R / F       ascend / descend
N / P       jump to next / previous highway corridor
SPACE       pause / resume auto-capture
C           force manual capture
ESC         quit and finalize session
```
