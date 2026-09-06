#!/usr/bin/env python3
"""
rs_phase3_weather_lighting.py
-----------------------------
RoadSentinel - Phase 3: Dynamic Time-of-Day and Weather System.
Supports both Unreal Engine 5.8 Python API and CARLA Python API.

Key features:
1. Weather preset library (8 presets: ClearNoon, OvercastDay, HeavyRain, LightDrizzle,
   DuskGoldenHour, NightClear, NightFoggy, NightRain).
2. Dynamic time-of-day and smooth preset interpolation over configurable duration (--transition-time).
3. Non-destructive parameter mutations on existing scene actors (DirectionalLight, SkyLight,
   ExponentialHeightFog, BP_Sky_Sphere); missing actors spawned only if absent and tagged RS_Weather.
4. Wet-surface material parameter RS_RoadWetness applied to road meshes and RS_Defect actors.
5. Nighttime bloom post-process volume PP_RS_NightBloom with RS_BloomIntensity = 2.0, threshold 0.9.
6. Road-level exponential height fog visibility ~60m (falloff 0.12) for NightFoggy/NightRain.
7. Validation logging to env/output/logs/rs_weather_transitions.jsonl.
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

# Ensure workspace root in path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Graceful imports
try:
    import carla
    HAS_CARLA = True
except ImportError:
    carla = None
    HAS_CARLA = False

try:
    import unreal
    HAS_UNREAL = True
except ImportError:
    unreal = None
    HAS_UNREAL = False


PRESETS_FILE = WORKSPACE_ROOT / "env" / "config" / "rs_weather_presets.json"
TRANSITIONS_LOG = WORKSPACE_ROOT / "env" / "output" / "logs" / "rs_weather_transitions.jsonl"


# ==============================================================================
# Weather Preset Data & Loader
# ==============================================================================

DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "ClearNoon": {
        "name": "ClearNoon",
        "description": "Sun altitude 70°, no fog, dry road roughness 0.85",
        "sun_altitude_deg": 70.0,
        "sun_azimuth_deg": 180.0,
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "road_wet_roughness": 0.85,
        "road_wetness": 0.0,
        "fog_density": 0.0,
        "fog_distance_m": 10000.0,
        "fog_falloff": 0.05,
        "visibility_m": 10000.0,
        "solar_intensity_lux": 100000.0,
        "color_temperature_k": 6500.0,
        "moonlight_lux": 0.0,
        "wind_intensity": 10.0,
        "streetlamp_emissive": false if 'false' in locals() else False,
        "splash_particles": False,
        "bloom_enabled": False,
        "bloom_intensity": 0.0,
        "lens_flare_threshold": 1.0,
        "time_of_day_normalized": 0.5
    },
    "OvercastDay": {
        "name": "OvercastDay",
        "description": "Sun altitude 40°, cloud cover 0.8, diffuse sky light",
        "sun_altitude_deg": 40.0,
        "sun_azimuth_deg": 160.0,
        "cloudiness": 80.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "road_wet_roughness": 0.80,
        "road_wetness": 0.0,
        "fog_density": 0.005,
        "fog_distance_m": 2500.0,
        "fog_falloff": 0.06,
        "visibility_m": 2500.0,
        "solar_intensity_lux": 35000.0,
        "color_temperature_k": 6500.0,
        "moonlight_lux": 0.0,
        "wind_intensity": 20.0,
        "streetlamp_emissive": False,
        "splash_particles": False,
        "bloom_enabled": False,
        "bloom_intensity": 0.0,
        "lens_flare_threshold": 1.0,
        "time_of_day_normalized": 0.38
    },
    "HeavyRain": {
        "name": "HeavyRain",
        "description": "Precipitation rate 80 mm/h, road wet roughness 0.25, splash particles",
        "sun_altitude_deg": 35.0,
        "sun_azimuth_deg": 150.0,
        "cloudiness": 95.0,
        "precipitation": 80.0,
        "precipitation_deposits": 1.0,
        "road_wet_roughness": 0.25,
        "road_wetness": 1.0,
        "fog_density": 0.02,
        "fog_distance_m": 400.0,
        "fog_falloff": 0.08,
        "visibility_m": 400.0,
        "solar_intensity_lux": 15000.0,
        "color_temperature_k": 6000.0,
        "moonlight_lux": 0.0,
        "wind_intensity": 65.0,
        "streetlamp_emissive": False,
        "splash_particles": True,
        "bloom_enabled": False,
        "bloom_intensity": 0.0,
        "lens_flare_threshold": 1.0,
        "time_of_day_normalized": 0.42
    },
    "LightDrizzle": {
        "name": "LightDrizzle",
        "description": "Precipitation rate 5 mm/h, road wet roughness 0.55",
        "sun_altitude_deg": 45.0,
        "sun_azimuth_deg": 140.0,
        "cloudiness": 65.0,
        "precipitation": 5.0,
        "precipitation_deposits": 0.45,
        "road_wet_roughness": 0.55,
        "road_wetness": 0.45,
        "fog_density": 0.01,
        "fog_distance_m": 1200.0,
        "fog_falloff": 0.06,
        "visibility_m": 1200.0,
        "solar_intensity_lux": 45000.0,
        "color_temperature_k": 6200.0,
        "moonlight_lux": 0.0,
        "wind_intensity": 25.0,
        "streetlamp_emissive": False,
        "splash_particles": False,
        "bloom_enabled": False,
        "bloom_intensity": 0.0,
        "lens_flare_threshold": 1.0,
        "time_of_day_normalized": 0.45
    },
    "DuskGoldenHour": {
        "name": "DuskGoldenHour",
        "description": "Sun altitude 5°, warm colour temperature 3200 K, long shadows",
        "sun_altitude_deg": 5.0,
        "sun_azimuth_deg": 260.0,
        "cloudiness": 30.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "road_wet_roughness": 0.85,
        "road_wetness": 0.0,
        "fog_density": 0.008,
        "fog_distance_m": 3500.0,
        "fog_falloff": 0.07,
        "visibility_m": 3500.0,
        "solar_intensity_lux": 25000.0,
        "color_temperature_k": 3200.0,
        "moonlight_lux": 0.0,
        "wind_intensity": 15.0,
        "streetlamp_emissive": False,
        "splash_particles": False,
        "bloom_enabled": False,
        "bloom_intensity": 0.0,
        "lens_flare_threshold": 1.0,
        "time_of_day_normalized": 0.75
    },
    "NightClear": {
        "name": "NightClear",
        "description": "Sun altitude −15°, moonlight 0.08 lux, streetlamp emissive enabled",
        "sun_altitude_deg": -15.0,
        "sun_azimuth_deg": 0.0,
        "cloudiness": 15.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "road_wet_roughness": 0.85,
        "road_wetness": 0.0,
        "fog_density": 0.002,
        "fog_distance_m": 6000.0,
        "fog_falloff": 0.04,
        "visibility_m": 6000.0,
        "solar_intensity_lux": 0.0,
        "color_temperature_k": 4000.0,
        "moonlight_lux": 0.08,
        "wind_intensity": 10.0,
        "streetlamp_emissive": True,
        "splash_particles": False,
        "bloom_enabled": True,
        "bloom_intensity": 0.5,
        "lens_flare_threshold": 0.95,
        "time_of_day_normalized": 0.0
    },
    "NightFoggy": {
        "name": "NightFoggy",
        "description": "Sun altitude −20°, exponential fog density 0.04, visibility 60 m",
        "sun_altitude_deg": -20.0,
        "sun_azimuth_deg": 0.0,
        "cloudiness": 70.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.2,
        "road_wet_roughness": 0.70,
        "road_wetness": 0.2,
        "fog_density": 0.04,
        "fog_distance_m": 60.0,
        "fog_falloff": 0.12,
        "visibility_m": 60.0,
        "solar_intensity_lux": 0.0,
        "color_temperature_k": 4200.0,
        "moonlight_lux": 0.04,
        "wind_intensity": 8.0,
        "streetlamp_emissive": True,
        "splash_particles": False,
        "bloom_enabled": True,
        "bloom_intensity": 2.0,
        "lens_flare_threshold": 0.9,
        "time_of_day_normalized": 0.08
    },
    "NightRain": {
        "name": "NightRain",
        "description": "Combines NightClear precipitation + HeavyRain wet road + headlight glare bloom",
        "sun_altitude_deg": -18.0,
        "sun_azimuth_deg": 0.0,
        "cloudiness": 95.0,
        "precipitation": 80.0,
        "precipitation_deposits": 1.0,
        "road_wet_roughness": 0.25,
        "road_wetness": 1.0,
        "fog_density": 0.035,
        "fog_distance_m": 60.0,
        "fog_falloff": 0.12,
        "visibility_m": 60.0,
        "solar_intensity_lux": 0.0,
        "color_temperature_k": 4500.0,
        "moonlight_lux": 0.03,
        "wind_intensity": 55.0,
        "streetlamp_emissive": True,
        "splash_particles": True,
        "bloom_enabled": True,
        "bloom_intensity": 2.0,
        "lens_flare_threshold": 0.9,
        "time_of_day_normalized": 0.12
    }
}


def load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Loads presets from env/config/rs_weather_presets.json.
    If absent, creates it with DEFAULT_PRESETS.
    If present, appends missing presets only without overwriting existing.
    """
    PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not PRESETS_FILE.exists():
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PRESETS, f, indent=2)
        return dict(DEFAULT_PRESETS)

    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        loaded = {}

    changed = False
    for k, v in DEFAULT_PRESETS.items():
        if k not in loaded:
            loaded[k] = v
            changed = True

    if changed:
        with open(PRESETS_FILE, "w", encoding="utf-8") as f:
            json.dump(loaded, f, indent=2)

    return loaded


# ==============================================================================
# Preset Interpolation & Time-of-Day Math
# ==============================================================================

def interpolate_presets(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    alpha: float
) -> Dict[str, Any]:
    """Linearly blends two preset dictionaries by factor alpha [0.0 - 1.0]."""
    alpha = max(0.0, min(1.0, float(alpha)))
    blended = dict(p1)

    numeric_keys = [
        "sun_altitude_deg", "sun_azimuth_deg", "cloudiness", "precipitation",
        "precipitation_deposits", "road_wet_roughness", "road_wetness",
        "fog_density", "fog_distance_m", "fog_falloff", "visibility_m",
        "solar_intensity_lux", "color_temperature_k", "moonlight_lux",
        "wind_intensity", "bloom_intensity", "lens_flare_threshold"
    ]

    for k in numeric_keys:
        v1 = float(p1.get(k, 0.0))
        v2 = float(p2.get(k, 0.0))
        blended[k] = round(v1 + alpha * (v2 - v1), 4)

    blended["streetlamp_emissive"] = p2["streetlamp_emissive"] if alpha >= 0.5 else p1["streetlamp_emissive"]
    blended["splash_particles"] = p2["splash_particles"] if alpha >= 0.5 else p1["splash_particles"]
    blended["bloom_enabled"] = p2["bloom_enabled"] if alpha >= 0.5 else p1["bloom_enabled"]
    blended["name"] = f"Blend_{p1['name']}_{p2['name']}_{int(alpha*100)}"
    return blended


def get_preset_by_time_of_day(
    time_of_day: float,
    presets: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes a blended preset given time_of_day in [0.0 - 1.0]
    (0.0 = midnight, 0.5 = noon, 0.75 = dusk, 1.0 = midnight).
    Automatically identifies the two nearest presets and interpolates.
    """
    tod = time_of_day % 1.0
    timeline = []
    for p in presets.values():
        timeline.append((p.get("time_of_day_normalized", 0.5), p))

    timeline.sort(key=lambda x: x[0])

    # Find bounding presets
    p_prev = timeline[-1][1]
    p_next = timeline[0][1]
    t_prev = timeline[-1][0] - 1.0
    t_next = timeline[0][0]

    for i in range(len(timeline)):
        t_curr, p_curr = timeline[i]
        if t_curr <= tod:
            p_prev = p_curr
            t_prev = t_curr
        else:
            p_next = p_curr
            t_next = t_curr
            break

    span = t_next - t_prev
    if span <= 1e-5:
        alpha = 0.0
    else:
        alpha = (tod - t_prev) / span

    return interpolate_presets(p_prev, p_next, alpha)


# ==============================================================================
# Unreal Engine Lighting Scene Controller
# ==============================================================================

class UnrealWeatherController:
    """
    Controls dynamic weather and lighting directly in Unreal Engine 5.8:
    - Queries existing DirectionalLight, SkyLight, ExponentialHeightFog, BP_Sky_Sphere.
    - Preserves existing scene hierarchy; drives properties via non-destructive mutations.
    - Spawns missing actors tagged RS_Weather only if absent.
    - Updates scalar parameter RS_RoadWetness across road and RS_Defect meshes.
    - Manages PP_RS_NightBloom post-process volume for headlight glare bloom.
    """

    def __init__(self):
        self.sun_light = None
        self.sky_light = None
        self.fog_actor = None
        self.sky_sphere = None
        self.bloom_pp = None
        self._initialized = False

    def initialize_actors(self):
        if not HAS_UNREAL or self._initialized:
            return

        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()

        for actor in all_actors:
            if not actor:
                continue
            actor_class = actor.get_class().get_name()
            actor_name = actor.get_actor_label()

            if "DirectionalLight" in actor_class or "DirectionalLight" in actor_name:
                if not self.sun_light:
                    self.sun_light = actor

            elif "SkyLight" in actor_class or "SkyLight" in actor_name:
                if not self.sky_light:
                    self.sky_light = actor

            elif "ExponentialHeightFog" in actor_class or "Fog" in actor_name:
                if not self.fog_actor:
                    self.fog_actor = actor

            elif "SkySphere" in actor_class or "BP_Sky_Sphere" in actor_name:
                if not self.sky_sphere:
                    self.sky_sphere = actor

            elif "PP_RS_NightBloom" in actor_name or (actor.actor_has_tag(unreal.Name("RS_Weather")) and "PostProcess" in actor_class):
                if not self.bloom_pp:
                    self.bloom_pp = actor

        # Spawn missing actors tagged RS_Weather
        if not self.sun_light:
            unreal.log("Spawning DirectionalLight with tag RS_Weather...")
            self.sun_light = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(0, 0, 0)
            )
            if self.sun_light:
                self.sun_light.tags.append(unreal.Name("RS_Weather"))
                self.sun_light.set_actor_label("RS_DirectionalLight")

        if not self.fog_actor:
            unreal.log("Spawning ExponentialHeightFog with tag RS_Weather...")
            self.fog_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.ExponentialHeightFog, unreal.Vector(0, 0, 100), unreal.Rotator(0, 0, 0)
            )
            if self.fog_actor:
                self.fog_actor.tags.append(unreal.Name("RS_Weather"))
                self.fog_actor.set_actor_label("RS_ExponentialHeightFog")

        if not self.bloom_pp:
            unreal.log("Spawning PP_RS_NightBloom PostProcessVolume with tag RS_Weather...")
            self.bloom_pp = unreal.EditorLevelLibrary.spawn_actor_from_class(
                unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0)
            )
            if self.bloom_pp:
                self.bloom_pp.tags.append(unreal.Name("RS_Weather"))
                self.bloom_pp.set_actor_label("PP_RS_NightBloom")
                self.bloom_pp.set_editor_property("unbound", True)

        self._initialized = True
        unreal.log("[RoadSentinel] Weather scene actors initialized successfully.")

    def apply_preset(self, preset: Dict[str, Any]):
        if not HAS_UNREAL:
            return

        self.initialize_actors()

        # 1. Drive Sun DirectionalLight (Pitch = altitude, Yaw = azimuth)
        if self.sun_light:
            alt = preset.get("sun_altitude_deg", 45.0)
            az = preset.get("sun_azimuth_deg", 180.0)
            # Pitch in Unreal: negative pitch points downwards
            new_rot = unreal.Rotator(alt, az, 0.0)
            self.sun_light.set_actor_rotation(new_rot, False)

            sun_comp = self.sun_light.get_component_by_class(unreal.DirectionalLightComponent)
            if sun_comp:
                lux = preset.get("solar_intensity_lux", 50000.0)
                sun_comp.set_editor_property("intensity", max(0.01, lux / 10.0))
                temp = preset.get("color_temperature_k", 6500.0)
                sun_comp.set_editor_property("use_temperature", True)
                sun_comp.set_editor_property("temperature", temp)

        # 2. Drive SkyLight
        if self.sky_light:
            sky_comp = self.sky_light.get_component_by_class(unreal.SkyLightComponent)
            if sky_comp:
                cloudiness = preset.get("cloudiness", 20.0)
                sky_intensity = 1.0 + (cloudiness / 100.0) * 2.5
                sky_comp.set_editor_property("intensity", sky_intensity)

        # 3. Drive ExponentialHeightFog (Atmospheric fog on road anomalies)
        if self.fog_actor:
            fog_comp = self.fog_actor.get_component_by_class(unreal.ExponentialHeightFogComponent)
            if fog_comp:
                density = preset.get("fog_density", 0.005)
                falloff = preset.get("fog_falloff", 0.08)
                fog_comp.set_editor_property("fog_density", density)
                fog_comp.set_editor_property("fog_height_falloff", falloff)

        # 4. Drive Nighttime Bloom (PP_RS_NightBloom)
        if self.bloom_pp:
            bloom_enabled = preset.get("bloom_enabled", False)
            bloom_intensity = preset.get("bloom_intensity", 2.0)
            lens_flare = preset.get("lens_flare_threshold", 0.9)
            settings = self.bloom_pp.get_editor_property("settings")
            if settings:
                settings.set_editor_property("override_bloom_intensity", True)
                settings.set_editor_property("bloom_intensity", bloom_intensity if bloom_enabled else 0.0)
                settings.set_editor_property("override_bloom_threshold", True)
                settings.set_editor_property("bloom_threshold", lens_flare)
                self.bloom_pp.set_editor_property("settings", settings)

        # 5. Wet-surface material response (RS_RoadWetness)
        self.update_road_wetness(preset.get("road_wetness", 0.0))

    def update_road_wetness(self, wetness: float):
        """Updates RS_RoadWetness on all road and RS_Defect tagged meshes without respawning."""
        if not HAS_UNREAL:
            return

        wetness = max(0.0, min(1.0, float(wetness)))
        all_actors = unreal.EditorLevelLibrary.get_all_level_actors()

        for actor in all_actors:
            if not actor:
                continue
            is_defect = actor.actor_has_tag(unreal.Name("RS_Defect"))
            name = actor.get_actor_label().lower()
            is_road = "road" in name or "highway" in name or "asphalt" in name

            if is_defect or is_road:
                comp = actor.get_component_by_class(unreal.StaticMeshComponent)
                if comp:
                    num_mats = comp.get_num_materials()
                    for idx in range(num_mats):
                        mid = comp.get_material(idx)
                        if isinstance(mid, unreal.MaterialInstanceDynamic):
                            mid.set_scalar_parameter_value("RS_RoadWetness", wetness)


# ==============================================================================
# CARLA Weather Wrapper
# ==============================================================================

class CarlaWeatherWrapper:
    """Controls weather and lighting in CARLA simulation via carla.WeatherParameters."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout_s: float = 5.0):
        self.client: Optional["carla.Client"] = None
        self.world: Optional["carla.World"] = None
        self.is_connected = False
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def connect(self) -> bool:
        if not HAS_CARLA:
            return False
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout_s)
            self.world = self.client.get_world()
            self.is_connected = True
            return True
        except Exception:
            self.is_connected = False
            return False

    def apply_preset(self, preset: Dict[str, Any]):
        if not self.is_connected and not self.connect():
            return

        try:
            wp = carla.WeatherParameters(
                cloudiness=float(preset.get("cloudiness", 0.0)),
                precipitation=float(preset.get("precipitation", 0.0)),
                precipitation_deposits=float(preset.get("precipitation_deposits", 0.0)),
                wind_intensity=float(preset.get("wind_intensity", 10.0)),
                sun_azimuth_angle=float(preset.get("sun_azimuth_deg", 180.0)),
                sun_altitude_angle=float(preset.get("sun_altitude_deg", 45.0)),
                fog_density=float(preset.get("fog_density", 0.0) * 100.0),
                fog_distance=float(preset.get("fog_distance_m", 1000.0)),
                fog_falloff=float(preset.get("fog_falloff", 0.1)),
                wetness=float(preset.get("road_wetness", 0.0) * 100.0)
            )
            self.world.set_weather(wp)
        except Exception as e:
            print(f"[RoadSentinel] CARLA weather update error: {e}")


# ==============================================================================
# Unified Weather & Lighting Manager
# ==============================================================================

class DynamicWeatherManager:
    """
    Unified manager handling:
    - Preset selection & interpolation
    - Real-time transition interpolation over configurable duration
    - Simultaneous dispatch to Unreal Engine scene & CARLA simulator
    - Transition history logging to rs_weather_transitions.jsonl
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2000):
        self.presets = load_presets()
        self.active_preset: Dict[str, Any] = self.presets.get("ClearNoon", DEFAULT_PRESETS["ClearNoon"])
        self.unreal_controller = UnrealWeatherController()
        self.carla_controller = CarlaWeatherWrapper(host, port)

    def set_preset(
        self,
        preset_name: str,
        transition_time_s: float = 10.0,
        log_transition: bool = True
    ) -> Dict[str, Any]:
        """Applies a named preset with smooth transition over transition_time_s."""
        target = self.presets.get(preset_name)
        if not target:
            raise ValueError(f"Unknown preset: '{preset_name}'. Valid: {list(self.presets.keys())}")

        return self.transition_to(target, transition_time_s, log_transition)

    def set_time_of_day(
        self,
        time_of_day: float,
        transition_time_s: float = 2.0,
        log_transition: bool = True
    ) -> Dict[str, Any]:
        """Blends to the preset matching time_of_day (0.0=midnight, 0.5=noon)."""
        target = get_preset_by_time_of_day(time_of_day, self.presets)
        return self.transition_to(target, transition_time_s, log_transition)

    def transition_to(
        self,
        target_preset: Dict[str, Any],
        duration_s: float = 10.0,
        log_transition: bool = True
    ) -> Dict[str, Any]:
        """Smoothly interpolates active preset to target_preset over duration_s."""
        start_preset = dict(self.active_preset)
        duration_s = max(0.01, float(duration_s))

        # Determine step count: ~10 steps per second (min 1 step)
        steps = max(1, int(round(duration_s * 10)))
        step_dt = duration_s / steps

        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            blended = interpolate_presets(start_preset, target_preset, alpha)
            self._dispatch_preset(blended)
            if duration_s > 0.1 and i < steps:
                time.sleep(step_dt)

        self.active_preset = target_preset
        self._dispatch_preset(target_preset)

        if log_transition:
            self._log_transition(target_preset)

        return target_preset

    def _dispatch_preset(self, preset: Dict[str, Any]):
        """Dispatches preset parameters to both Unreal and CARLA controllers."""
        if HAS_UNREAL:
            self.unreal_controller.apply_preset(preset)
        if HAS_CARLA and self.carla_controller.is_connected:
            self.carla_controller.apply_preset(preset)

    def _log_transition(self, preset: Dict[str, Any]):
        """Appends active parameter bundle to env/output/logs/rs_weather_transitions.jsonl."""
        TRANSITIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "timestamp_ms": int(time.time() * 1000),
            "preset_name": preset.get("name", "Custom"),
            "parameters": preset
        }
        with open(TRANSITIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Dynamic Weather & Lighting System")
    parser.add_argument("--preset", type=str, default=None, choices=list(DEFAULT_PRESETS.keys()),
                        help="Named weather preset to activate")
    parser.add_argument("--transition-time", type=float, default=10.0,
                        help="Transition duration in seconds (default: 10.0)")
    parser.add_argument("--time-of-day", type=float, default=None,
                        help="Normalized time of day 0.0=midnight to 0.5=noon")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="CARLA host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port")
    parser.add_argument("--list-presets", action="store_true", help="List all available presets and exit")

    args, unknown = parser.parse_known_args()

    presets = load_presets()

    if args.list_presets:
        print("\nAvailable RoadSentinel Weather Presets:")
        for name, data in presets.items():
            print(f"  - {name:16s}: {data.get('description', '')}")
        return

    manager = DynamicWeatherManager(host=args.host, port=args.port)

    if args.time_of_day is not None:
        print(f"Setting time of day to {args.time_of_day:.2f} over {args.transition_time}s...")
        res = manager.set_time_of_day(args.time_of_day, transition_time_s=args.transition_time)
        print(f"Active condition: {res['name']}")

    elif args.preset:
        print(f"Transitioning to preset '{args.preset}' over {args.transition_time}s...")
        res = manager.set_preset(args.preset, transition_time_s=args.transition_time)
        print(f"Active condition: {res['name']}")

    else:
        # Default to ClearNoon
        print("No preset specified. Transitioning to 'ClearNoon'...")
        manager.set_preset("ClearNoon", transition_time_s=1.0)


if __name__ == "__main__":
    main()
