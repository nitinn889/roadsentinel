"""Regression tests for persistent 20-day Unreal temporal experiment setup."""

import json

import temporal_20_day as temporal

from temporal_20_day import (
    EVALUATION_START_DAY,
    SEGMENTS,
    build_day_manifest,
    pick_segment_frame,
    predict_held_out,
)


def test_temporal_manifest_keeps_segment_ids_and_coordinates_stable():
    day_1 = build_day_manifest(1, seed=2026)
    day_20 = build_day_manifest(20, seed=2026)

    assert len(day_1["segments"]) == len(SEGMENTS)
    assert [s["road_segment_id"] for s in day_1["segments"]] == [s["road_segment_id"] for s in day_20["segments"]]
    assert [(s["along_m"], s["across_m"]) for s in day_1["segments"]] == [
        (s["along_m"], s["across_m"]) for s in day_20["segments"]
    ]

    # The rapid segment begins healthy and first manifests as a visible crack;
    # once visible its defect ID/location must remain stable as it worsens.
    rapid_early = next(d for d in build_day_manifest(3, seed=2026)["defects"] if d["road_segment_id"] == "SEG_004")
    rapid_late = next(d for d in day_20["defects"] if d["road_segment_id"] == "SEG_004")
    assert rapid_early["defect_id"] == rapid_late["defect_id"]
    assert rapid_early["along_m"] == rapid_late["along_m"]
    assert rapid_late["true_severity_score"] > rapid_early["true_severity_score"]
    assert day_1["unreal_capture"]["source"] == "Unreal SceneCapture2D render target"


def test_unreal_capture_selection_uses_the_explicit_persistent_segment_id(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "SEG_001.png").write_bytes(b"png")
    (image_dir / "SEG_002.png").write_bytes(b"png")
    rows = [
        {"road_segment_id": "SEG_001", "image_name": "SEG_001.png", "sim_time_s": "999"},
        {"road_segment_id": "SEG_002", "image_name": "SEG_002.png", "sim_time_s": "0"},
    ]

    image, row = pick_segment_frame(tmp_path, rows, SEGMENTS[0])

    assert image.name == "SEG_001.png"
    assert row["road_segment_id"] == "SEG_001"


def test_unreal_batch_uses_the_editor_capture_worker_not_carla(tmp_path, monkeypatch):
    project = tmp_path / "RoadSentinelSim.uproject"
    launcher = tmp_path / "launch_unreal_editor.sh"
    worker = tmp_path / "rs_temporal_20_day_capture.py"
    for path in (project, launcher, worker):
        path.write_text("placeholder", encoding="utf-8")

    manifest = tmp_path / "manifests" / "day_01.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    capture_dir = tmp_path / "captures" / "day_01"
    request_path = tmp_path / "unreal_capture_request.json"
    monkeypatch.setattr(temporal, "UNREAL_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(temporal, "UNREAL_PROJECT", project)
    monkeypatch.setattr(temporal, "UNREAL_LAUNCHER", launcher)
    monkeypatch.setattr(temporal, "UNREAL_CAPTURE_SCRIPT", worker)

    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        image_dir = capture_dir / "images"
        image_dir.mkdir(parents=True)
        for segment in SEGMENTS:
            (image_dir / f"{segment.segment_id}.png").write_bytes(b"png")
        (capture_dir / "metadata.csv").write_text("road_segment_id,image_name\n", encoding="utf-8")
        (capture_dir / "ground_truth.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(temporal.subprocess, "run", fake_run)
    temporal.run_unreal_capture_batch([(1, manifest, capture_dir)], request_path)

    assert "-RenderOffscreen" in observed["command"]
    assert any("rs_temporal_20_day_capture.py" in part for part in observed["command"])
    assert all("carla" not in part.lower() for part in observed["command"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["entries"][0]["capture_dir"] == str(capture_dir.resolve())


def test_temporal_prediction_excludes_final_six_days_from_fit():
    history = [{"day": day, "severity_score": day / 30.0} for day in range(1, 21)]
    result = predict_held_out(history)

    assert result["training_days"] == list(range(1, EVALUATION_START_DAY))
    assert result["unseen_evaluation_days"] == list(range(EVALUATION_START_DAY, 21))
    assert len(result["held_out_predictions"]) == 6
    assert 0.0 <= result["predicted_future_severity"] <= 1.0
