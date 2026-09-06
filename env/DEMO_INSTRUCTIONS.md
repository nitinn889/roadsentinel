# RoadSentinel Drone Simulation — Complete Demo-Day Setup & Startup Guide

This document is the complete demo-day runbook for the RoadSentinel drone simulation, updated to match the setup that was actually tested and working on the development laptop.

The goal is to start from a powered-on laptop, launch CARLA, verify the CARLA Python connection, start the RoadSentinel drone simulator, demonstrate flight and image capture, and collect the generated output files.

---

## 1. Current Tested Setup

### Project location

```text
/home/nitin-nandakumar/Downloads/roadsentinel/
```

Important folders:

```text
roadsentinel/
├── env/                         # RoadSentinel drone simulation code
│   ├── config.py
│   ├── drone_controller.py
│   ├── drone_sim.py
│   └── requirements.txt
├── carla_env/                   # Python 3.10 environment for CARLA client
├── ml/                          # ML pipeline
├── road_health_pipeline/        # Road-health pipeline
└── RoadSentinel_datasets/       # Datasets
```

### CARLA version

```text
CARLA 0.9.16
```

CARLA is being run from the Docker image:

```text
carlasim/carla:0.9.16
```

### Python environment

Use:

```text
Python 3.10.21
```

Virtual environment:

```text
~/Downloads/roadsentinel/carla_env
```

Do **not** use the project's `.venv` for the CARLA simulator. The `.venv` uses Python 3.14 and is not suitable for the CARLA 0.9.16 Python client used here.

### Current tested simulation settings

```text
Map:              Town04
Drone altitude:   50 m
Flight speed:     30 km/h (8.333 m/s)
Camera FOV:       100 degrees
Live camera:      640 × 360
Target overlap:   70%
```

The 640 × 360 camera is the currently tested stable configuration for smooth simulation. Higher-resolution capture is a planned improvement described later in this document.

---

# 2. What Runs During the Demo

There are two separate components:

### A. CARLA simulator/server

CARLA is the Unreal Engine-based simulation server. It provides the virtual road environment, physics, world, and camera sensor.

It runs in one terminal and must remain running throughout the demonstration.

### B. RoadSentinel Python client

`drone_sim.py` is the RoadSentinel client. It connects to CARLA over:

```text
127.0.0.1:2000
```

It controls the simulated drone, moves the camera, displays the live preview, and saves captured images and metadata.

The CARLA server must be running **before** `drone_sim.py` is started.

---

# 3. Demo-Day Startup — From Turning on the Laptop

## Step 1 — Turn on the laptop and wait for Ubuntu to load

Log into the normal Ubuntu desktop environment.

Do not open the drone simulator yet.

---

## Step 2 — Check that Docker is running

Open **Terminal 1**.

Run:

```bash
docker --version
```

Then:

```bash
docker ps
```

Docker should respond normally.

If Docker gives a daemon/permission error, start the Docker service:

```bash
sudo systemctl start docker
```

Then check again:

```bash
docker ps
```

---

## Step 3 — Verify the CARLA 0.9.16 image is available

Run:

```bash
docker images | grep carla
```

You should see the CARLA image:

```text
carlasim/carla    0.9.16
```

If the image is already present, do not download it again.

---

## Step 4 — Activate the correct Python environment

In Terminal 1, run:

```bash
cd ~/Downloads/roadsentinel
source carla_env/bin/activate
```

The terminal prompt should begin with:

```text
(carla_env)
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.10.21
```

---

## Step 5 — Verify the CARLA Python API

Run:

```bash
python -c "import carla; print(carla.Client)"
```

Expected output will look like:

```text
<class 'carla.libcarla.Client'>
```

This confirms that the RoadSentinel Python environment can import the CARLA client library.

---

# 4. Start the CARLA Simulator

## Step 6 — Prepare the graphical environment

Before starting CARLA, run:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
```

Then:

```bash
xhost +local:docker
```

This allows the Dockerized CARLA application to display its Unreal Engine window on the Ubuntu desktop.

---

## Step 7 — Make sure an old CARLA instance is not occupying port 2000

Run:

```bash
sudo ss -ltnp | grep :2000
```

### Normal result

Nothing should be printed if no CARLA server is currently running.

### If an old CARLA process is present

Stop it:

```bash
sudo pkill -9 -f CarlaUE4
```

Then verify again:

```bash
sudo ss -ltnp | grep :2000
```

The port should now be free.

**Only one CARLA server should be running on port 2000.**

---

## Step 8 — Start CARLA 0.9.16

In **Terminal 1**, run:

```bash
docker run --rm -it \
  --gpus all \
  --runtime=nvidia \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -opengl -quality-level=Low -nosound
```

Leave this terminal running.

The **CARLA UE4 window** should open.

Do not close the UE4 window during the demo.

### Important

The currently reliable configuration uses:

```text
-opengl
-quality-level=Low
-nosound
```

Do not add `-RenderOffScreen` when using the visible UE4 window for this demo unless the setup has been separately tested. In the tested setup, the visible UE4 window is used as the CARLA server/renderer.

---

# 5. Verify CARLA Before Running RoadSentinel

## Step 9 — Open Terminal 2

Open a new terminal window.

Activate the CARLA environment:

```bash
cd ~/Downloads/roadsentinel
source carla_env/bin/activate
```

---

## Step 10 — Test the CARLA server connection

Run:

```bash
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); print('SERVER:',c.get_server_version()); print('MAP:',c.get_world().get_map().name)"
```

A healthy server should return immediately with something like:

```text
SERVER: 0.9.16
MAP: Carla/Maps/...
```

Do not start `drone_sim.py` if this command times out.

---

# 6. Set the RoadSentinel Simulation Configuration

## Step 11 — Open the configuration file

The configuration file is:

```text
~/Downloads/roadsentinel/env/config.py
```

The current intended demo configuration is:

```python
CARLA_MAP = "Town04"
ALTITUDE_M = 50.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
```

The other important settings are:

```text
Speed:          30 km/h
FOV:            100 degrees
Overlap:        70%
```

Check the values with:

```bash
cd ~/Downloads/roadsentinel/env
grep -nE "CARLA_MAP|ALTITUDE_M|IMAGE_WIDTH|IMAGE_HEIGHT|CAMERA_FOV_DEG" config.py
```

### Why 640 × 360?

The 640 × 360 camera configuration was the first configuration that produced a stable, responsive live simulation after the higher-resolution camera caused CARLA synchronous ticks to stall on this laptop setup.

The live preview resolution is intentionally low so the flight demonstration remains smooth.

---

# 7. Start the RoadSentinel Drone Simulator

## Step 12 — Run the Python simulator

In **Terminal 2**:

```bash
cd ~/Downloads/roadsentinel/env
python drone_sim.py
```

A successful startup should print a flight/capture parameter block similar to:

```text
--- Flight / capture parameters ---
Speed:            30.0 km/h (8.333 m/s), constant
Altitude:         50.0 m
Camera FOV (H):   100.0 deg, 640x360
Target overlap:   70%
------------------------------------
Found ... candidate straight road segment(s). Use N/P to cycle.
Setup complete. Focus the preview window and fly. ESC to quit and finalize output.
```

The exact ground footprint/GSD/capture interval depends on the configuration in `config.py`.

---

# 8. What the Person Watching the Demo Should See

At this point there should be two visible components:

### CARLA UE4 window

This is the simulated world/server.

### RoadSentinel Drone Preview window

This is the operator-facing control window. It shows the drone camera feed and receives the keyboard controls.

**Click inside the RoadSentinel Drone Preview window before pressing flight controls.**

Otherwise the keyboard may still be focused on another window.

---

# 9. Drone Controls

The current controls are:

```text
W / S       Forward / backward
A / D       Strafe left / right
Q / E       Yaw left / right
R / F       Ascend / descend
N / P       Next / previous straight-road segment
SPACE       Pause / resume automatic photo capture
C           Force an immediate photo capture
ESC         Stop the simulation and finalize output
```

### Recommended demo sequence

1. Click the RoadSentinel preview window.
2. Use `N` or `P` to select a suitable straight road segment.
3. Use `W` to begin forward movement.
4. Demonstrate altitude adjustment with `R`/`F` if appropriate.
5. Press `C` once to demonstrate manual image capture.
6. Allow automatic capture to continue while flying along the road.
7. After a useful section of road has been inspected, press `ESC`.

---

# 10. Ending the Demonstration Correctly

**Do not close the terminal to stop the simulator.**

Press:

```text
ESC
```

inside the RoadSentinel preview window.

This allows the script to:

- stop the camera sensor,
- destroy the CARLA actor,
- restore the CARLA world settings,
- finish writing metadata,
- finalize the capture log,
- close the Pygame preview cleanly.

You should see a completion message similar to:

```text
Done. N images + geo.txt + metadata.csv written to .../output
```

---

# 11. Check the Captured Output

The expected output directory is:

```text
~/Downloads/roadsentinel/env/output/
```

Check it with:

```bash
cd ~/Downloads/roadsentinel/env
ls -lh output/
```

Expected files/folders include:

```text
output/
├── images/
├── geo.txt
├── metadata.csv
└── capture_log.json
```

Check captured images:

```bash
ls -lh output/images/
```

Open the most recent image:

```bash
xdg-open "$(ls -t output/images/* | head -1)"
```

---

# 12. What Each Output File Represents

## `output/images/`

Sequentially captured RGB road photographs.

These are the images intended for the next stage of the RoadSentinel pipeline.

## `geo.txt`

Geotag-style output containing the longitude, latitude, and altitude associated with captured images.

## `metadata.csv`

Per-photo metadata such as camera pose, timestamp, and other capture information.

## `capture_log.json`

Run-level information such as configuration and image count.

---

# 13. Pre-Demo Quick Test

The recommended dry run is:

1. Turn on the laptop.
2. Open Terminal 1.
3. Confirm Docker is running.
4. Activate `carla_env`.
5. Start the CARLA 0.9.16 Docker server.
6. Wait for the CARLA UE4 window.
7. Open Terminal 2.
8. Verify `SERVER: 0.9.16` with the CARLA connection test.
9. Verify the configured map is available.
10. Run `python drone_sim.py`.
11. Confirm the RoadSentinel preview opens.
12. Click the preview window.
13. Press `C` and confirm an image is captured.
14. Fly for approximately 30–60 seconds.
15. Press `ESC`.
16. Confirm `output/images/` contains images.
17. Confirm `metadata.csv` exists.
18. Open one captured image to verify it is not black or corrupted.

---

# 14. Troubleshooting

## Problem: `Could not connect to CARLA at 127.0.0.1:2000`

Check whether CARLA is running:

```bash
sudo ss -ltnp | grep :2000
```

If nothing appears, the CARLA server is not running.

Start the CARLA Docker command again.

---

## Problem: Port 2000 is already in use

Find the process:

```bash
sudo lsof -i :2000
```

If it is an old `CarlaUE4` process:

```bash
sudo pkill -9 -f CarlaUE4
```

Then confirm the port is free:

```bash
sudo ss -ltnp | grep :2000
```

Start exactly one CARLA server afterward.

---

## Problem: `ModuleNotFoundError: No module named 'carla'`

Check the active environment:

```bash
which python
python --version
```

It should point to:

```text
~/Downloads/roadsentinel/carla_env/bin/python
```

and show Python 3.10.x.

Do not run the drone simulator from the Python 3.14 `.venv` environment.

---

## Problem: `ModuleNotFoundError: No module named 'pygame'` or `numpy`

Activate the correct environment:

```bash
cd ~/Downloads/roadsentinel
source carla_env/bin/activate
```

Then install the simulation requirements:

```bash
pip install pygame numpy
```

The project's requirements file contains an older CARLA version (`0.9.14`), so do **not** blindly reinstall CARLA from that file. The CARLA Python client being used for this demo is 0.9.16.

---

## Problem: `pip install carla==0.9.14` fails

This is expected with the current environment/package combination.

The project requirements originally referenced CARLA 0.9.14, while the working server/client setup for this demo is CARLA 0.9.16.

The matching CARLA 0.9.16 wheel used is the Python 3.10 build:

```text
carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

Do not downgrade the server to 0.9.14 just to satisfy the old requirements file.

---

## Problem: CARLA exits immediately during Vulkan startup

The Docker image may show messages involving Vulkan or `lavapipe`.

For the tested visible-window configuration, use OpenGL:

```text
-opengl
```

with:

```text
-quality-level=Low
```

and:

```text
-nosound
```

The tested server command is given in Section 4.

---

## Problem: `XDG_RUNTIME_DIR not set`

Run on the host:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
```

Then start CARLA using the Docker command in Section 4, including:

```text
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR
-v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR
```

---

## Problem: RoadSentinel preview opens but is blank/unresponsive

First verify that the CARLA server responds to:

```bash
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); print(c.get_server_version()); print(c.get_world().get_map().name)"
```

If the command times out, CARLA itself is not responding correctly and `drone_sim.py` should not be started again until that is fixed.

If CARLA responds but the simulation stalls during `world.tick()`, use the stable lower camera resolution:

```python
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
```

---

## Problem: Simulator is too slow

Use:

```text
-quality-level=Low
```

and keep the live RGB sensor at:

```text
640 × 360
```

The visible UE4 renderer should also remain on OpenGL for the current tested setup.

Do not immediately return to 4K if the goal is a smooth live demonstration. High-resolution capture is a separate improvement described below.

---

## Problem: `Warning: no frame received from camera this tick.`

This means the camera sensor did not deliver its frame during the synchronous world tick.

First reduce the camera resolution:

```text
640 × 360
```

Then restart `drone_sim.py`.

If necessary, restart the CARLA server as well so the simulation begins from a clean state.

---

## Problem: The map is not changing

Changing:

```python
CARLA_MAP = "Town04"
```

in `config.py` does not itself change the CARLA server's current world.

Verify the actual server map with:

```bash
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); print(c.get_world().get_map().name)"
```

If the server still reports:

```text
Carla/Maps/Town10HD_Opt
```

then the currently running CARLA instance is still on Town10HD_Opt.

For the demo, the server and configuration should both be aligned with the intended map before starting the drone.

---

# 15. Road Environment / Rural Scene Notes

The ideal RoadSentinel demonstration environment is a rural or secondary-road environment because the real system is intended for road-condition inspection rather than simply demonstrating highway navigation.

The standard CARLA Town07 environment is known for rural-style scenery, but **Town07 is not present in the currently installed CARLA 0.9.16 package used for this demo**.

The installed map list was verified and contains:

```text
Town01
Town01_Opt
Town02
Town02_Opt
Town03
Town03_Opt
Town04
Town04_Opt
Town05
Town05_Opt
Town10HD
Town10HD_Opt
```

For the current demo build, **Town04** is the selected available alternative.

If a future CARLA installation includes Town07, it would be a better rural scene choice for RoadSentinel.

---

# 16. Potholes in the Current CARLA Scene

The standard CARLA Town maps do not automatically provide a set of RoadSentinel-specific potholes for this project.

Therefore, simply changing from Town10HD_Opt to Town04 does not automatically create pothole targets.

For the final RoadSentinel demonstration, the scene should include deliberately placed road defects such as:

```text
Small pothole
Medium pothole
Large/severe pothole
Water-filled pothole
Cluster of multiple potholes
```

These can then be captured by the simulated drone and used to demonstrate the downstream pothole detection and severity-analysis pipeline.

---

# 17. Planned Improvements

## Improvement 1 — High-resolution image capture with low-resolution live preview

The current stable setup uses a low-resolution RGB camera so that the simulator remains responsive.

The intended final design is:

```text
Live preview camera       → 640 × 360
Saved RoadSentinel RGB    → 1920 × 1080 or 3840 × 2160
```

The purpose is to keep the operator's live view smooth while giving the ML pipeline higher-quality road imagery.

A future implementation can use separate preview and capture camera sensors, or otherwise separate the live display resolution from the saved capture resolution.

The high-resolution camera should be tested carefully because CARLA still has to render the sensor internally; increasing saved-image resolution can reduce simulation performance even when the Pygame window itself remains small.

---

## Improvement 2 — Add potholes to the simulated road

Create visible artificial potholes in CARLA so the drone can perform an actual road-inspection demonstration rather than flying over defect-free roads.

Recommended test cases:

```text
1. Small shallow pothole
2. Medium pothole
3. Large/deep pothole
4. Water-filled pothole
5. Multiple potholes close together
```

The different defect types can be used to demonstrate detection, segmentation, localization, severity scoring, and the water-hazard rule in the RoadSentinel pipeline.

---

## Improvement 3 — Lower the drone altitude

The current tested altitude is:

```text
50 m
```

A future test should reduce altitude further, for example:

```text
30–40 m
```

The purpose is to increase road-surface detail and make potholes, cracks, water, and road texture occupy more pixels in the RGB images.

Altitude should be selected together with camera FOV, sensor resolution, flight speed, and overlap so that the resulting coverage and image spacing remain appropriate.

---

# 18. Recommended Final Demo Flow

When the system is fully stabilized, the ideal presentation sequence is:

```text
Laptop powered on
        ↓
Docker ready
        ↓
CARLA 0.9.16 started
        ↓
Rural/secondary-road map loaded
        ↓
RoadSentinel Python client connected
        ↓
Drone starts at planned altitude
        ↓
Live low-resolution preview shown
        ↓
Drone follows road segment
        ↓
Automatic overlapping image capture
        ↓
High-resolution RGB images saved
        ↓
Potholes visible in captured imagery
        ↓
Images passed to RoadSentinel ML pipeline
        ↓
Potholes segmented/detected
        ↓
Severity and geolocation information generated
        ↓
Road-health results visualized
```

---

# 19. Five-Minute Demo Checklist

Before the demo starts, confirm:

- [ ] Laptop connected to power.
- [ ] NVIDIA GPU is available.
- [ ] Docker is running.
- [ ] CARLA image `carlasim/carla:0.9.16` exists.
- [ ] No old CARLA process is occupying port 2000.
- [ ] `carla_env` is active.
- [ ] Python reports 3.10.x.
- [ ] `import carla` works.
- [ ] CARLA UE4 window is open.
- [ ] CARLA responds to the 127.0.0.1:2000 connection test.
- [ ] The intended map is loaded.
- [ ] `config.py` has the intended map and altitude.
- [ ] Live camera is at the tested stable resolution.
- [ ] `drone_sim.py` starts successfully.
- [ ] RoadSentinel preview window is visible.
- [ ] Keyboard focus is on the preview window.
- [ ] `C` successfully captures an image.
- [ ] Automatic capture is working.
- [ ] `ESC` finalizes the run.
- [ ] `output/images/` contains captured images.
- [ ] `metadata.csv` is present.

---

# 20. Essential Commands — Quick Reference

## Start Dockerized CARLA

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
xhost +local:docker

sudo ss -ltnp | grep :2000

# Only if an old CARLA process exists:
sudo pkill -9 -f CarlaUE4

docker run --rm -it \
  --gpus all \
  --runtime=nvidia \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh -opengl -quality-level=Low -nosound
```

## Verify CARLA

In Terminal 2:

```bash
cd ~/Downloads/roadsentinel
source carla_env/bin/activate
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(20); print('SERVER:',c.get_server_version()); print('MAP:',c.get_world().get_map().name)"
```

## Start RoadSentinel

```bash
cd ~/Downloads/roadsentinel/env
python drone_sim.py
```

## Check output

```bash
cd ~/Downloads/roadsentinel/env
ls -lh output/
ls -lh output/images/
```

---

# 21. Demo-Day Rule to Remember

**Do not start multiple CARLA servers.**

The correct order is always:

```text
1. Start one CARLA 0.9.16 server.
2. Wait for the UE4 simulator to be ready.
3. Verify 127.0.0.1:2000 responds.
4. Start RoadSentinel `drone_sim.py`.
5. Fly and capture.
6. Press ESC to finalize.
```

If CARLA is not responding, fix CARLA first. Do not repeatedly restart `drone_sim.py` against a stalled or partially initialized CARLA server.
