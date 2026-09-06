# Nitin — Blender Road & Pothole Simulation — Context Sheet

## How to use this document
- 5 parts, self-contained — paste each part's full prompt into a fresh AI chat (with Blender's Python API knowledge, e.g. Claude/ChatGPT) to get working `bpy` scripts.
- Complete in order: Part 1 → Part 5.
- If a session runs out mid-part, start fresh, paste the same part again, and add: *"I already have [X] working — continue from there,"* pasting your current script.
- **Your job produces the INPUT that Marion's AI pipeline consumes.** The exact output format in Part 4 and Part 5 is a strict contract — don't change field names without telling Marion and Vrinda.

---

## PART 1 — Blender Scene Setup + Procedural Road Generation

```
I'm building the simulation component of "RoadSentinel" — a drone-based pothole
detection and predictive road-health monitoring system. My specific job is to build
a Blender scene (using Python/bpy scripting, not manual modeling) that generates
realistic road surfaces with randomly placed potholes, then simulates a drone flying
over and photographing it — producing synthetic training/testing data for a
teammate's AI pipeline (DINOv2 + SAM2 based detection).

WHY SIMULATION: We can't test on real public roads (safety/access constraints), and
constructing real potholes for testing is impractical. Blender gives us a second big
advantage: since WE define every pothole's true depth/area/location, we get perfect
ground-truth data to validate the AI pipeline's accuracy against — something
impossible with real-world data.

YOUR TASK FOR THIS PART:

1. Write a Python script (to run inside Blender, using `bpy`) that procedurally
   generates a flat road surface as a subdivided plane mesh (e.g. 20m x 100m,
   subdivided finely enough, e.g. 0.05m per subdivision, to allow later mesh
   deformation for potholes).

2. Apply a realistic asphalt-like material: a procedural shader using noise
   textures for a granular, non-uniform grey/dark surface (not a flat solid color) —
   this matters because DINOv2's anomaly detection depends on textural realism, not
   just geometry.

3. Add subtle random surface roughness across the WHOLE road (small-amplitude noise
   displacement) to simulate that even "healthy" road isn't perfectly flat — this
   is important so the healthy-road reference isn't trivially different from a
   pothole region by flatness alone.

4. Set up basic scene lighting (a sun lamp with adjustable angle/strength) so I can
   later vary lighting conditions between scenes.

5. Wrap all of this in a reusable Python function `generate_base_road(length_m,
   width_m, subdivision_size_m, seed)` so I can call it repeatedly with different
   parameters/seeds to produce varied road scenes.

Write clean, well-commented bpy code I can run directly in Blender's scripting tab
or via `blender --background --python script.py`. Explain briefly what each bpy
operation does, since I'm still learning the API, but prioritize working code.
```

---

## PART 2 — Procedural Pothole Generation (Random, Including Water-Filled)

```
I'm building the simulation component of "RoadSentinel" (drone-based pothole
detection). [Paste Part 1's context block here if starting a fresh session.]

WHAT'S ALREADY BUILT (Part 1):
- `generate_base_road(length_m, width_m, subdivision_size_m, seed)` — produces a
  road mesh with realistic asphalt material and subtle base-level surface noise.

YOUR TASK FOR THIS PART:

1. Write a function `generate_potholes(road_mesh, num_potholes, seed, water_ratio=0.25)`
   that procedurally creates a given number of pothole-like depressions on the road
   mesh, each with RANDOMIZED properties for realism and dataset diversity:
   - Random position on the road surface (avoiding overlaps with each other)
   - Random shape/size: irregular (not perfectly circular) — use noise-perturbed
     circular/elliptical vertex selection, not a perfect boolean circle
   - Random depth (e.g. uniformly sampled between 1cm and 8cm — shallow to severe)
   - Random diameter (e.g. 10cm to 60cm)
   - Depression created via actual mesh vertex displacement (push selected vertices
     downward with falloff toward the rim, like a real bowl-shaped pothole), NOT a
     texture-only fake — this matters because later we simulate photogrammetry
     depth recovery, which needs real 3D geometry to reconstruct.
   - Rougher/more broken-looking material or vertex noise INSIDE the pothole
     region specifically (real potholes look more fragmented/gravelly than
     surrounding road) to give the anomaly detector a genuine texture cue.

2. For a configurable fraction of potholes (`water_ratio` parameter, e.g. 25%), add
   a WATER FILL: a flat, semi-transparent, reflective plane object positioned at a
   random height between the pothole's true floor and its rim (simulating partial
   filling), with a water-like material (high reflectivity, slight blue-grey
   tint, smooth/glossy — NOT the rough material used for dry pothole floors). This
   is critical for testing the water-hazard detection logic later.

3. As each pothole is generated, RECORD its ground-truth properties in a Python
   list/dict: unique ID, true center position (local scene X/Y coordinates), true
   max depth (cm), true diameter/area (cm²), and whether it's water-filled
   (and if so, the true water-surface depth vs. true floor depth, since these
   differ — this is exactly what we need to validate the refraction-correction
   math later).

4. Wrap this into the reusable function signature above, callable with different
   seeds to produce many distinct scenes with different pothole layouts.

Write clean, well-commented bpy code extending Part 1. Explain key steps, but
prioritize working code I can run directly.
```

---

## PART 3 — Simulated Drone Camera Rig (Nadir Waypoint Mission)

```
I'm building the simulation component of "RoadSentinel" (drone-based pothole
detection). [Paste Part 1's context block here if starting a fresh session.]

WHAT'S ALREADY BUILT (Parts 1-2):
- A road mesh with realistic material and base surface noise (Part 1).
- Procedurally generated potholes (dry and water-filled) with recorded ground-truth
  properties (Part 2).

YOUR TASK FOR THIS PART:

Simulate a real drone's behavior: flying a straight waypoint path at a FIXED,
KNOWN altitude, camera facing straight down (nadir), capturing photos at regular
intervals along the path with 70-80% forward overlap between consecutive photos
(this overlap is essential later for the AI pipeline's photogrammetric depth
reconstruction).

1. Write a function `setup_nadir_camera(altitude_m, sensor_width_mm, focal_length_mm)`
   that creates a Blender camera object, positions it at the given altitude above
   the road, and rotates it to point straight down (-Z direction). Set the camera's
   sensor width and focal length to realistic consumer-drone-camera values (explain
   what values you're choosing and why, e.g. matching a typical DJI camera spec) —
   these exact values matter later for the Ground Sample Distance calculation in
   Marion's pipeline, so keep them as clearly labeled, easy-to-find constants.

2. Write a function `compute_waypoint_positions(road_length_m, altitude_m,
   sensor_width_mm, focal_length_mm, image_width_px, overlap_ratio=0.75)` that
   calculates how far apart (in meters along the road) consecutive camera
   positions should be to achieve the target overlap ratio, given the camera's
   field of view at that altitude. Show the math clearly.

3. Write a function `render_waypoint_sequence(output_dir, road_length_m, altitude_m,
   camera_params, image_resolution=(1920,1080))` that:
   - Moves the camera to each computed waypoint position in turn
   - Renders a photo at each position (using Blender's rendering, e.g. Cycles or
     Eevee — pick whichever is faster for this flat, simple scene and explain why)
   - Saves each rendered image to `output_dir` with a clear sequential filename
     (e.g. `frame_0001.png`, `frame_0002.png`, ...)

4. Add slight, realistic RANDOM VARIATION to each waypoint capture to avoid
   unrealistically perfect data: small random camera position jitter (a few cm),
   small random tilt jitter (a degree or two off perfect nadir), and optionally
   randomized sun angle/lighting per scene — real drone flights are never perfectly
   smooth, and the AI pipeline needs to be tested against that realism.

Write clean, well-commented bpy code extending Parts 1-2. Explain the overlap-
distance math clearly since it matters for correctness, but prioritize working code.
```

---

## PART 4 — Simulated Geotagging Metadata (Standing in for Real EXIF GPS)

```
I'm building the simulation component of "RoadSentinel" (drone-based pothole
detection). [Paste Part 1's context block here if starting a fresh session.]

WHAT'S ALREADY BUILT (Parts 1-3):
- Road + procedurally generated potholes (dry and water-filled) with ground-truth
  properties.
- A nadir camera rig that flies a waypoint path and renders overlapping photos to
  disk.

WHY THIS PART MATTERS: real drone photos carry GPS location automatically in their
EXIF metadata. Blender obviously can't produce real GPS coordinates, so we need to
simulate this by mapping Blender's local scene coordinates to a plausible fake
GPS coordinate system, in a way that's consistent and usable by the AI pipeline
exactly like real EXIF data would be.

YOUR TASK FOR THIS PART:

1. Write a function `scene_xy_to_fake_gps(x_m, y_m, origin_lat, origin_lon)` that
   converts a position in the Blender scene (meters along/across the road) into a
   plausible fake latitude/longitude, using a simple local flat-earth approximation
   (small-scale conversion: roughly 111,320 meters per degree of latitude, and
   longitude scaled by cos(latitude) — explain this briefly). Pick a reasonable
   arbitrary origin_lat/origin_lon (e.g. somewhere generic) as a constant.

2. Modify (or extend) the Part 3 rendering function so that for EVERY rendered
   frame, it also writes a matching metadata JSON file (same base filename, e.g.
   `frame_0001.json` next to `frame_0001.png`) containing:
   ```
   {
     "frame_id": "frame_0001",
     "sim_gps_lat": <float>,
     "sim_gps_lon": <float>,
     "altitude_m": <float>,
     "timestamp": "<ISO8601 string, can be synthetic sequential timestamps>",
     "image_path": "frame_0001.png"
   }
   ```
   This is the exact format Marion's pipeline expects in place of real EXIF data —
   don't change field names.

3. Write a small helper that also converts each pothole's ground-truth center
   position (from Part 2) into the same fake GPS coordinate system, so ground-truth
   locations can later be directly compared against the AI pipeline's detected
   locations (same coordinate space, apples-to-apples).

Write clean, well-commented code extending Parts 1-3. Explain the coordinate
conversion math briefly, but prioritize working code that produces correctly
formatted JSON files.
```

---

## PART 5 — Ground-Truth Export + Multi-Scene Batch Generation

```
I'm building the simulation component of "RoadSentinel" (drone-based pothole
detection). [Paste Part 1's context block here if starting a fresh session.]

WHAT'S ALREADY BUILT (Parts 1-4):
- Full single-scene pipeline: generate road + potholes (dry/water) with ground
  truth, fly a simulated nadir waypoint mission, render overlapping photos, and
  write matching fake-GPS metadata JSON per photo.

YOUR TASK FOR THIS PART (final part):

1. Write a function `export_ground_truth(pothole_records, output_path)` that saves
   ALL the ground-truth pothole data recorded during generation (Part 2 + Part 4's
   GPS conversion) to a single `ground_truth.json` file per scene, one entry per
   pothole:
   ```
   {
     "pothole_id": "<string>",
     "true_area_cm2": <float>,
     "true_max_depth_cm": <float>,
     "true_gps_lat": <float>,
     "true_gps_lon": <float>,
     "is_water_filled": <bool>,
     "true_water_surface_depth_cm": <float or null>
   }
   ```
   This file is what Marion will use to check how accurate the AI pipeline's
   detected area/depth/water-flag values are, compared to what we KNOW is actually
   true in the simulation — this is our main validation tool before ever touching
   real drone data.

2. Write a top-level function `generate_scene(scene_id, output_root_dir, seed,
   num_potholes, water_ratio, altitude_m, road_length_m, sun_angle=None)` that
   ties Parts 1-5 together: builds one complete randomized scene, renders its full
   waypoint photo sequence with metadata, and exports its ground truth — all saved
   under a clearly organized folder per scene, e.g.:
   ```
   output_root_dir/
     scene_001/
       frame_0001.png
       frame_0001.json
       frame_0002.png
       frame_0002.json
       ...
       ground_truth.json
   ```

3. Write a batch-generation script that calls `generate_scene()` multiple times
   with different seeds/parameters (varying pothole count, water ratio, lighting,
   road length) to produce a diverse SET of scenes in one run — this is what makes
   simulation genuinely valuable: we can cheaply generate far more test variety
   than would ever be practical to build physically.

4. Print a summary at the end of the batch run (e.g. "Generated 10 scenes, 87 total
   potholes, 22 water-filled, X photos total") so I have an at-a-glance confirmation
   of what was produced.

Write clean, well-commented code extending Parts 1-4, and make sure the final
output folder structure and JSON formats exactly match what's described above —
Marion and Vrinda's work depends on this format staying consistent.
```
