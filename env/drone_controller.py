"""
drone_controller.py
--------------------
Turns pygame keyboard state into a new carla.Transform for the drone each
tick. The drone has no physics/collision - it's a kinematic rig (see
README for why this is the normal way to do free-flying camera work in
CARLA) - so movement is just "add a displacement vector to the current
transform", with the horizontal speed magnitude clamped to a *constant*
30 km/h whenever any movement key is held (not a soft maximum - pressing
a direction key always yields exactly that speed, and diagonal input is
normalized so it can't exceed it). This constancy matters: the overlap
math in overlap_calculator.py assumes a fixed forward speed.

Controls
  W / S       forward / backward
  A / D       strafe left / right
  Q / E       yaw rotate left / right
  R / F       ascend / descend (within configured altitude band)
  N / P       jump to next / previous straight road segment
  SPACE       pause / resume automatic photo capture
  C           force an immediate capture (manual, doesn't reset the timer... 
              actually it does reset the timer, see drone_sim.py)
  ESC         quit and finalize output
"""

import math
try:
    import carla
except ImportError:
    class _DummyLocation:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)
    class _DummyRotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)
    class _DummyTransform:
        def __init__(self, location=None, rotation=None):
            self.location = location or _DummyLocation()
            self.rotation = rotation or _DummyRotation()
    class DummyCarla:
        Location = _DummyLocation
        Rotation = _DummyRotation
        Transform = _DummyTransform
    carla = DummyCarla()

import pygame
import config


class DroneController:
    def __init__(self, initial_transform: "carla.Transform", speed_mps: float = config.SPEED_MPS):
        self.transform = initial_transform
        self.speed_mps = speed_mps
        self.paused = False
        self.road_segment_index = 0
        self.manual_capture_requested = False
        self.quit_requested = False

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_c:
                    self.manual_capture_requested = True
                elif event.key == pygame.K_n:
                    self.road_segment_index += 1
                elif event.key == pygame.K_p:
                    self.road_segment_index -= 1

    def update(self, dt: float, keys) -> "carla.Transform":
        """Advance self.transform by one tick given held-key state. Returns the new transform."""
        loc = self.transform.location
        rot = self.transform.rotation

        # --- yaw ---
        yaw_input = 0.0
        if keys[pygame.K_e]:
            yaw_input += 1.0
        if keys[pygame.K_q]:
            yaw_input -= 1.0
        rot.yaw += yaw_input * config.YAW_RATE_DEG_S * dt

        # --- horizontal movement (constant speed, normalized diagonal) ---
        forward_input = 0.0
        right_input = 0.0
        if keys[pygame.K_w]:
            forward_input += 1.0
        if keys[pygame.K_s]:
            forward_input -= 1.0
        if keys[pygame.K_d]:
            right_input += 1.0
        if keys[pygame.K_a]:
            right_input -= 1.0

        mag = math.hypot(forward_input, right_input)
        if mag > 1e-6:
            forward_input /= mag
            right_input /= mag

            yaw_rad = math.radians(rot.yaw)
            fwd_x, fwd_y = math.cos(yaw_rad), math.sin(yaw_rad)
            right_x, right_y = -math.sin(yaw_rad), math.cos(yaw_rad)

            dx = (fwd_x * forward_input + right_x * right_input) * self.speed_mps * dt
            dy = (fwd_y * forward_input + right_y * right_input) * self.speed_mps * dt
            loc.x += dx
            loc.y += dy

        # --- altitude ---
        climb_input = 0.0
        if keys[pygame.K_r]:
            climb_input += 1.0
        if keys[pygame.K_f]:
            climb_input -= 1.0
        if climb_input != 0.0:
            loc.z += climb_input * config.CLIMB_RATE_MPS * dt
            loc.z = max(config.ALTITUDE_MIN_M, min(config.ALTITUDE_MAX_M, loc.z))

        self.transform = carla.Transform(loc, rot)
        return self.transform

    def is_moving_horizontally(self, keys) -> bool:
        return any(keys[k] for k in (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d))
