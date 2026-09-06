#!/usr/bin/env python3
"""
test_phase4_dashboard.py
------------------------
Automated test suite verifying the Phase 4 Flask Blueprint and simulation controller endpoints:
1. GET  /rs/                   - Live dashboard HTML & SSE format
2. GET  /rs/control            - Parameter control form
3. POST /rs/control            - Pydantic RS_SimConfig validation (valid & invalid cases)
4. GET  /rs/stream             - MJPEG multipart stream generator
5. GET  /rs/annotations/latest - Ground-truth annotation JSON
6. GET  /rs/status             - Status telemetry, CARLA check, and disk space guard
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from env.app import app, RS_SimConfig, VALID_PRESETS

def test_all_endpoints():
    client = app.test_client()
    passed = 0
    total = 0

    print("=================================================================")
    print(" RoadSentinel Phase 4: Flask Dashboard & API Verification Suite   ")
    print("=================================================================\n")

    # 1. Root redirect
    total += 1
    res = client.get("/")
    assert res.status_code in [301, 302], f"Root should redirect, got {res.status_code}"
    print(f"[PASS] 1. GET / -> Redirects to /rs/ (Status {res.status_code})")
    passed += 1

    # 2. GET /rs/ (HTML dashboard)
    total += 1
    res = client.get("/rs/")
    assert res.status_code == 200, f"GET /rs/ failed: {res.status_code}"
    assert b"RoadSentinel" in res.data, "Dashboard title missing"
    assert b"val-weather-badge" in res.data, "Weather badge missing"
    print(f"[PASS] 2. GET /rs/ -> Live Dashboard HTML rendered (Status 200)")
    passed += 1

    # 3. GET /rs/?sse=1 (SSE event stream)
    total += 1
    res = client.get("/rs/?sse=1", headers={"Accept": "text/event-stream"})
    assert res.status_code == 200, f"SSE stream failed: {res.status_code}"
    assert "text/event-stream" in res.headers.get("Content-Type", ""), "MIME type should be text/event-stream"
    print(f"[PASS] 3. GET /rs/?sse=1 -> Telemetry SSE Stream verified (text/event-stream)")
    passed += 1

    # 4. GET /rs/control (Form)
    total += 1
    res = client.get("/rs/control")
    assert res.status_code == 200, f"GET /rs/control failed: {res.status_code}"
    assert b"input-weather-preset" in res.data, "Weather select element missing"
    assert b"input-defect-density" in res.data, "Defect density slider missing"
    assert b"input-altitude" in res.data, "Altitude slider missing"
    print(f"[PASS] 4. GET /rs/control -> Parameter form rendered (Status 200)")
    passed += 1

    # 5. POST /rs/control (Valid configuration)
    total += 1
    valid_payload = {
        "weather_preset": "DuskGoldenHour",
        "defect_density": 4.5,
        "defect_types": ["PotholeWet", "LongitudinalCrack"],
        "water_ior": 1.345,
        "altitude_m": 35.0,
        "flight_mode": "transect",
        "is_capturing": True,
        "transition_time_s": 1.0
    }
    res = client.post("/rs/control", json=valid_payload)
    assert res.status_code == 200, f"POST /rs/control failed: {res.status_code}, {res.data}"
    data = res.get_json()
    assert data["status"] == "success", f"Expected success: {data}"
    assert data["config"]["weather_preset"] == "DuskGoldenHour"
    assert data["config"]["altitude_m"] == 35.0
    print(f"[PASS] 5. POST /rs/control -> Valid config accepted & applied: {data['updated']}")
    passed += 1

    # 6. POST /rs/control (Validation rejection for out-of-range density)
    total += 1
    invalid_density = {"defect_density": 25.0}  # max is 10.0
    res = client.post("/rs/control", json=invalid_density)
    assert res.status_code == 400, f"Expected 400 validation error, got {res.status_code}"
    print(f"[PASS] 6. POST /rs/control -> Correctly rejected out-of-range density (Status 400)")
    passed += 1

    # 7. POST /rs/control (Validation rejection for invalid preset)
    total += 1
    invalid_preset = {"weather_preset": "NonExistentTornado"}
    res = client.post("/rs/control", json=invalid_preset)
    assert res.status_code == 400, f"Expected 400 for unknown preset, got {res.status_code}"
    print(f"[PASS] 7. POST /rs/control -> Correctly rejected invalid preset (Status 400)")
    passed += 1

    # 8. GET /rs/status (Telemetry and disk usage guard)
    total += 1
    res = client.get("/rs/status")
    assert res.status_code == 200, f"GET /rs/status failed: {res.status_code}"
    data = res.get_json()
    assert "carla_server_reachable" in data, "carla_server_reachable missing"
    assert "active_preset" in data, "active_preset missing"
    assert "drone_position" in data, "drone_position missing"
    assert "disk_usage" in data, "disk_usage missing"
    assert "free_gb" in data["disk_usage"], "disk_usage.free_gb missing"
    assert "warning" in data["disk_usage"], "disk_usage.warning missing"
    print(f"[PASS] 8. GET /rs/status -> Telemetry & disk guard verified:")
    print(f"       Preset: {data['active_preset']} | Pose: {data['drone_position']} | Free Disk: {data['disk_usage']['free_gb']} GB (Warning: {data['disk_usage']['warning']})")
    passed += 1

    # 9. GET /rs/annotations/latest
    total += 1
    res = client.get("/rs/annotations/latest")
    assert res.status_code == 200, f"GET /rs/annotations/latest failed: {res.status_code}"
    ann = res.get_json()
    assert "defects" in ann or "status" in ann, "Invalid annotation structure"
    print(f"[PASS] 9. GET /rs/annotations/latest -> JSON retrieved (Frame: {ann.get('frame_id', 'N/A')}, Defects: {ann.get('total_defects_in_frame', 0)})")
    passed += 1

    # 10. GET /rs/stream (MJPEG stream check)
    total += 1
    res = client.get("/rs/stream")
    assert res.status_code == 200, f"GET /rs/stream failed: {res.status_code}"
    assert "multipart/x-mixed-replace" in res.headers.get("Content-Type", ""), "MIME type should be multipart MJPEG"
    # Read first chunk
    chunk = next(res.response)
    assert b"--frame" in chunk, "MJPEG chunk should contain --frame boundary"
    assert b"image/jpeg" in chunk, "MJPEG chunk should specify image/jpeg"
    print(f"[PASS] 10. GET /rs/stream -> MJPEG stream verified with valid boundary chunk ({len(chunk)} bytes)")
    passed += 1

    print("\n=================================================================")
    print(f" ALL {passed}/{total} VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print("=================================================================\n")

if __name__ == "__main__":
    test_all_endpoints()
