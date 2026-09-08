#!/usr/bin/env python3
"""
rs_interactive_studio.py
------------------------
RoadSentinel - Interactive 3D Studio & Drone Controller GUI.
Runs via PySide6 and communicates seamlessly with Unreal Engine via local IPC socket.

Features:
- [NEW] Inspection Mode: Day 1-20 selector, Segment SEG_001-SEG_006 selector,
  Open/Generate Environment button, status display, Go-to-Segment, Capture Inspection.
- Atmospheric Lighting & Weather Dropdown
- Road Health & Pothole Degradation State Dropdown
- Pothole Sizing Spectrum (20cm to 1.6m)
- Scatter Density Slider (0.5 to 10.0 / 100m²)
- Pothole Moisture State (Wet Waterlogged vs Dry Rock)
- Instant Drone Camera Teleportation
- High-Resolution Photo Capture on 'C' Key (with camera pose logging)
- '🌍 GENERATE WORLD' button to dynamically rebuild the scene in Unreal Engine in real-time
"""

import os
import sys
import json
import time
import socket
from datetime import datetime, timezone
from pathlib import Path

from PySide6 import QtWidgets, QtCore, QtGui
from rs_capture_metadata_logger import record_capture_attempt

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = WORKSPACE_ROOT / "env" / "output" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_INSPECTIONS_ROOT = WORKSPACE_ROOT / "env" / "output" / "manual_inspections"
MANIFESTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_20_day" / "manifests"
TEMPORAL_SEGMENTS_ROOT = WORKSPACE_ROOT / "env" / "output" / "temporal_segments"

IPC_HOST = "127.0.0.1"
IPC_PORT = 8899

SEGMENT_IDS = [f"SEG_{n:03d}" for n in range(1, 7)]  # SEG_001 … SEG_006 (matching all manifests)


class IPCClient:
    """Handles communication with the Unreal Engine RoadSentinel server."""
    def __init__(self, host=IPC_HOST, port=IPC_PORT):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            return True
        except Exception:
            self.sock = None
            return False

    def send_command(self, payload: dict, timeout: float = 4.0) -> dict:
        if not self.sock:
            if not self.connect():
                return {"status": "error", "message": "Unreal Engine IPC server not connected."}
        try:
            msg = (json.dumps(payload) + "\n").encode("utf-8")
            self.sock.sendall(msg)
            self.sock.settimeout(timeout)
            resp_data = self.sock.recv(8192).decode("utf-8").strip()
            if resp_data:
                return json.loads(resp_data)
        except Exception as e:
            self.sock = None
            # Retry once
            if self.connect():
                try:
                    self.sock.sendall(msg)
                    resp_data = self.sock.recv(8192).decode("utf-8").strip()
                    if resp_data:
                        return json.loads(resp_data)
                except Exception:
                    pass
            return {"status": "error", "message": str(e)}
        return {"status": "ok"}


def _load_condition_score(day: int, segment_id: str) -> float:
    """Read the condition score from the local manifest JSON without IPC."""
    try:
        path = MANIFESTS_ROOT / f"day_{day:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for seg in payload.get("segments", []):
            if seg.get("road_segment_id") == segment_id:
                return float(seg.get("renderer_condition_score", 0.0))
    except Exception:
        pass
    return 0.0


class RoadSentinelStudioWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ipc = IPCClient()
        self.setWindowTitle("RoadSentinel 3D Studio & Drone Controller")
        self.resize(480, 920)
        self.setMinimumSize(440, 800)
        self._temporal_segments: dict = {}   # {segment_id: {...camera_pose...}}
        self._current_day: int = 1
        self._current_segment: str = SEGMENT_IDS[0]
        self.setup_styling()
        self.build_ui()
        self.check_connection()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    def setup_styling(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121418;
                color: #E2E8F0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #2D3748;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 18px;
                font-weight: bold;
                color: #63B3ED;
                background-color: #1A202C;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 10px;
                background-color: #2B6CB0;
                color: white;
                border-radius: 4px;
            }
            QGroupBox#groupInspection {
                border: 1px solid #38A169;
                background-color: #1A2020;
            }
            QGroupBox#groupInspection::title {
                background-color: #276749;
            }
            QLabel {
                color: #CBD5E0;
                font-size: 13px;
            }
            QLabel#statusPanel {
                background-color: #0D1117;
                color: #68D391;
                font-size: 12px;
                font-family: monospace;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #2D3748;
            }
            QComboBox {
                background-color: #2D3748;
                border: 1px solid #4A5568;
                border-radius: 6px;
                padding: 6px 12px;
                color: #EDF2F7;
                font-size: 13px;
            }
            QComboBox:hover {
                border: 1px solid #63B3ED;
            }
            QComboBox QAbstractItemView {
                background-color: #2D3748;
                selection-background-color: #3182CE;
                color: #EDF2F7;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #4A5568;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #3182CE;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #63B3ED;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QPushButton {
                background-color: #3182CE;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4299E1;
            }
            QPushButton:pressed {
                background-color: #2B6CB0;
            }
            QPushButton#btnGenerate {
                background-color: #38A169;
                font-size: 15px;
                padding: 12px 18px;
            }
            QPushButton#btnGenerate:hover {
                background-color: #48BB78;
            }
            QPushButton#btnCapture {
                background-color: #DD6B20;
                font-size: 15px;
                padding: 12px 18px;
            }
            QPushButton#btnCapture:hover {
                background-color: #ED8936;
            }
            QPushButton#btnOpenEnv {
                background-color: #276749;
                font-size: 14px;
                padding: 11px 16px;
            }
            QPushButton#btnOpenEnv:hover {
                background-color: #38A169;
            }
            QPushButton#btnOpenEnv:disabled {
                background-color: #2D3748;
                color: #718096;
            }
            QPushButton#btnGotoSegment {
                background-color: #2C5282;
                font-size: 13px;
                padding: 9px 14px;
            }
            QPushButton#btnGotoSegment:hover {
                background-color: #3182CE;
            }
            QPushButton#btnGotoSegment:disabled {
                background-color: #2D3748;
                color: #718096;
            }
            QPushButton#btnInspCapture {
                background-color: #B7791F;
                font-size: 14px;
                padding: 11px 16px;
            }
            QPushButton#btnInspCapture:hover {
                background-color: #D69E2E;
            }
            QPushButton#btnInspCapture:disabled {
                background-color: #2D3748;
                color: #718096;
            }
            QStatusBar {
                color: #A0AEC0;
                background-color: #0D1117;
                font-size: 12px;
            }
        """)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)
        scroll.setWidget(scroll_content)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ── Header ──────────────────────────────────────────────────────────
        title_label = QtWidgets.QLabel("🚁 RoadSentinel 3D Studio")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #63B3ED;")
        subtitle_label = QtWidgets.QLabel(
            "Hyper-Real Highway · Defect Generator · Inspection Capture"
        )
        subtitle_label.setStyleSheet("font-size: 12px; color: #A0AEC0; margin-bottom: 4px;")
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # ════════════════════════════════════════════════════════════════════
        # Section 0 — Master Inspection Controller (2 Selectors)
        # ════════════════════════════════════════════════════════════════════
        group_insp = QtWidgets.QGroupBox("0. 🔬 Master Inspection Controller")
        group_insp.setObjectName("groupInspection")
        layout_insp = QtWidgets.QVBoxLayout(group_insp)
        layout_insp.setSpacing(8)

        # Day / Segment selectors (side by side)
        row_selectors = QtWidgets.QHBoxLayout()

        col_seg = QtWidgets.QVBoxLayout()
        col_seg.addWidget(QtWidgets.QLabel("Road Segment:"))
        self.combo_segment = QtWidgets.QComboBox()
        self.combo_segment.setObjectName("comboSegment")
        self.combo_segment.addItems(SEGMENT_IDS)
        self.combo_segment.currentIndexChanged.connect(self._on_day_or_segment_changed)
        col_seg.addWidget(self.combo_segment)
        row_selectors.addLayout(col_seg)

        col_day = QtWidgets.QVBoxLayout()
        col_day.addWidget(QtWidgets.QLabel("Deterioration Day:"))
        self.combo_day = QtWidgets.QComboBox()
        self.combo_day.setObjectName("comboDay")
        for d in range(1, 11):
            self.combo_day.addItem(f"Day {d:02d}", userData=d)
        self.combo_day.currentIndexChanged.connect(self._on_day_or_segment_changed)
        col_day.addWidget(self.combo_day)
        row_selectors.addLayout(col_day)

        layout_insp.addLayout(row_selectors)

        # Materialise & Inspect button
        self.btn_open_env = QtWidgets.QPushButton("🛠  Materialise & Inspect")
        self.btn_open_env.setObjectName("btnOpenEnv")
        self.btn_open_env.setToolTip(
            "Materialise the exact deterministic road deterioration state in Unreal Engine\n"
            "and automatically position the camera for a top-down inspection."
        )
        self.btn_open_env.clicked.connect(self.on_open_env_clicked)
        layout_insp.addWidget(self.btn_open_env)

        # Status display panel
        self.lbl_insp_status = QtWidgets.QLabel("SEG_001 | Day 01 | Score: — | ⚪ Ready")
        self.lbl_insp_status.setObjectName("statusPanel")
        self.lbl_insp_status.setWordWrap(True)
        layout_insp.addWidget(self.lbl_insp_status)

        # Go-to-Segment + Capture row
        row_actions = QtWidgets.QHBoxLayout()
        self.btn_goto_segment = QtWidgets.QPushButton("📍 Re-align Camera")
        self.btn_goto_segment.setObjectName("btnGotoSegment")
        self.btn_goto_segment.setEnabled(True)
        self.btn_goto_segment.setToolTip(
            "Move the Unreal viewport camera to the fixed downward-facing\n"
            "inspection pose for the selected road segment."
        )
        self.btn_goto_segment.clicked.connect(self.on_goto_segment_clicked)
        row_actions.addWidget(self.btn_goto_segment)

        self.btn_insp_capture = QtWidgets.QPushButton("📸 Capture Inspection Frame [C]")
        self.btn_insp_capture.setObjectName("btnInspCapture")
        self.btn_insp_capture.setEnabled(True)
        self.btn_insp_capture.setToolTip(
            "Render top-down inspection capture and save to:\n"
            "env/output/temporal_segments/SEG_XXX/day_XX.png"
        )
        self.btn_insp_capture.clicked.connect(self.on_insp_capture_clicked)
        row_actions.addWidget(self.btn_insp_capture)
        layout_insp.addLayout(row_actions)

        # Hint about saved path
        self.lbl_insp_saved = QtWidgets.QLabel("")
        self.lbl_insp_saved.setStyleSheet("color: #68D391; font-size: 11px;")
        self.lbl_insp_saved.setWordWrap(True)
        layout_insp.addWidget(self.lbl_insp_saved)

        main_layout.addWidget(group_insp)

        # ════════════════════════════════════════════════════════════════════
        # Section 1 — Atmospheric Lighting & Weather (unchanged)
        # ════════════════════════════════════════════════════════════════════
        group_light = QtWidgets.QGroupBox("1. 🌤 Atmospheric Lighting & Weather")
        layout_light = QtWidgets.QVBoxLayout(group_light)

        layout_light.addWidget(QtWidgets.QLabel("Lighting Preset:"))
        self.combo_light = QtWidgets.QComboBox()
        self.combo_light.addItems([
            "Clear Noon (70° Sun)",
            "Golden Hour Sunset",
            "Overcast Day",
            "Heavy Rain & Wet Road",
            "Dense Atmospheric Fog",
            "Night Highway with Lamps"
        ])
        self.combo_light.currentIndexChanged.connect(self.on_lighting_changed)
        layout_light.addWidget(self.combo_light)
        main_layout.addWidget(group_light)

        # ════════════════════════════════════════════════════════════════════
        # Section 2 — Road Health & Defect Parameters (unchanged)
        # ════════════════════════════════════════════════════════════════════
        group_road = QtWidgets.QGroupBox("2. 🛣 Road Health & Pothole Sizing")
        layout_road = QtWidgets.QVBoxLayout(group_road)

        layout_road.addWidget(QtWidgets.QLabel("Road Degradation State:"))
        self.combo_health = QtWidgets.QComboBox()
        self.combo_health.addItems([
            "Pristine (Grade A)",
            "Minor Wear (Grade B)",
            "Moderate Deterioration (Grade C)",
            "Severe Breakdown (Grade D - Critical)",
            "Critical Hazard (Grade F)"
        ])
        self.combo_health.setCurrentIndex(2)  # Moderate
        layout_road.addWidget(self.combo_health)

        layout_road.addWidget(QtWidgets.QLabel("Pothole Sizing Spectrum:"))
        self.combo_sizing = QtWidgets.QComboBox()
        self.combo_sizing.addItems([
            "Multi-Scale Organic (20cm - 1.6m)",
            "Large Severe Craters Only",
            "Micro Pitting & Hairlines"
        ])
        layout_road.addWidget(self.combo_sizing)

        # Density Slider
        density_header = QtWidgets.QHBoxLayout()
        density_header.addWidget(QtWidgets.QLabel("Pothole Scatter Density:"))
        self.lbl_density = QtWidgets.QLabel("3.5 / 100m²")
        self.lbl_density.setStyleSheet("color: #63B3ED; font-weight: bold;")
        density_header.addWidget(self.lbl_density, alignment=QtCore.Qt.AlignRight)
        layout_road.addLayout(density_header)

        self.slider_density = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_density.setRange(5, 100)  # 0.5 to 10.0
        self.slider_density.setValue(35)
        self.slider_density.valueChanged.connect(self.on_density_changed)
        layout_road.addWidget(self.slider_density)

        # Water puddle ratio
        layout_road.addWidget(QtWidgets.QLabel("Pothole Moisture State:"))
        self.combo_water = QtWidgets.QComboBox()
        self.combo_water.addItems([
            "Mixed Wet / Dry Cavities",
            "100% Waterlogged Puddles",
            "100% Dry Crushed Aggregate"
        ])
        layout_road.addWidget(self.combo_water)
        main_layout.addWidget(group_road)

        # ════════════════════════════════════════════════════════════════════
        # Section 3 — Drone Navigation & Photo Capture (unchanged)
        # ════════════════════════════════════════════════════════════════════
        group_drone = QtWidgets.QGroupBox("3. 🚁 Drone Flight & Capture ('C' Key)")
        layout_drone = QtWidgets.QVBoxLayout(group_drone)

        layout_drone.addWidget(QtWidgets.QLabel("Teleport Drone Camera View:"))
        self.combo_viewpoint = QtWidgets.QComboBox()
        self.combo_viewpoint.addItems([
            "🔭 Overhead Drone Survey (SAM 2 Top-Down)",
            "🔍 Low-Angle Pothole Inspection (30° Close-Up)",
            "💧 Waterlogged Pothole Macro View",
            "🛣 Highway Cruise View (Forward Drone)",
            "🌄 Highway Curve Vantage Overlook"
        ])
        layout_drone.addWidget(self.combo_viewpoint)

        btn_teleport = QtWidgets.QPushButton("🚀 Teleport Camera")
        btn_teleport.clicked.connect(self.on_teleport_clicked)
        layout_drone.addWidget(btn_teleport)

        # Big Capture Photo Button
        self.btn_capture = QtWidgets.QPushButton("📸 CAPTURE PHOTO (Press 'C')")
        self.btn_capture.setObjectName("btnCapture")
        self.btn_capture.clicked.connect(self.on_capture_clicked)
        layout_drone.addWidget(self.btn_capture)
        main_layout.addWidget(group_drone)

        # ════════════════════════════════════════════════════════════════════
        # Section 4 — Main Action (unchanged)
        # ════════════════════════════════════════════════════════════════════
        self.btn_generate = QtWidgets.QPushButton("🌍 GENERATE WORLD")
        self.btn_generate.setObjectName("btnGenerate")
        self.btn_generate.clicked.connect(self.on_generate_clicked)
        main_layout.addWidget(self.btn_generate)

        main_layout.addStretch()

        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Select Day & Segment → Open Environment → Capture.")

        # Initialise status panel text
        self._refresh_status_panel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_day(self) -> int:
        return self.combo_day.currentData() or 1

    def _selected_segment(self) -> str:
        return self.combo_segment.currentText()

    def _inspection_metadata_snapshot(self) -> dict:
        """Read, without changing, the exact GUI values shown at capture time."""
        return {
            "segment_id": self.combo_segment.currentText(),
            "day": int(self.combo_day.currentData() or 1),
            "day_display": self.combo_day.currentText(),
            "lighting_preset": self.combo_light.currentText(),
            "road_health_state": self.combo_health.currentText(),
            "pothole_sizing_spectrum": self.combo_sizing.currentText(),
            "pothole_density_per_100m2": self.slider_density.value() / 10.0,
            "pothole_density_display": self.lbl_density.text(),
            "pothole_moisture_state": self.combo_water.currentText(),
            "camera_preset": self.combo_viewpoint.currentText(),
            "capture_timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        }

    def _safe_inspection_metadata_snapshot(self):
        """Never allow optional metadata reads to interrupt image capture."""
        try:
            return self._inspection_metadata_snapshot()
        except Exception as exc:
            print(f"Warning: inspection metadata snapshot failed: {exc}", file=sys.stderr)
            return None

    def _log_inspection_metadata_passively(self, snapshot: dict, response: dict) -> None:
        """Write metadata without allowing failures to affect image capture."""
        try:
            fallback = (
                TEMPORAL_SEGMENTS_ROOT
                / snapshot["segment_id"]
                / f"day_{snapshot['day']:02d}.png"
            )
            response_path = response.get("path")
            image_path = Path(response_path) if response_path else fallback
            if not image_path.is_absolute():
                image_path = WORKSPACE_ROOT / image_path
            status = str(response.get("status", "unknown"))
            record_capture_attempt(
                snapshot,
                image_path=image_path,
                output_root=TEMPORAL_SEGMENTS_ROOT,
                workspace_root=WORKSPACE_ROOT,
                capture_succeeded=status in ("ok", "warning"),
                capture_status=status,
            )
        except Exception as exc:
            print(f"Warning: inspection metadata logging failed: {exc}", file=sys.stderr)

    def _refresh_status_panel(self, extra: str = ""):
        day = self._selected_day()
        seg = self._selected_segment()
        score = _load_condition_score(day, seg)
        loaded = bool(self._temporal_segments)
        cam_status = "🟢 Scene loaded" if loaded else "⚪ Not loaded"
        lines = [
            f"Day: {day:02d}   Segment: {seg}",
            f"Condition Score: {score:.4f}",
            f"Camera: {cam_status}",
        ]
        if extra:
            lines.append(extra)
        self.lbl_insp_status.setText("\n".join(lines))

    def check_connection(self):
        if self.ipc.connect():
            self.status.showMessage("✓ Connected to Unreal Engine 5 Editor.")
        else:
            self.status.showMessage(
                "⚠ Unreal Engine IPC not detected. "
                "Run rs_phase2_surface_defects.py in the UE Python console first."
            )

    # ------------------------------------------------------------------
    # Inspection Capture Mode Slots (NEW)
    # ------------------------------------------------------------------

    def _on_day_or_segment_changed(self):
        """Update status panel whenever day or segment selection changes."""
        self._current_day = self._selected_day()
        self._current_segment = self._selected_segment()
        # If a scene is loaded, the segments dict may be for a different day —
        # clear it so the user must re-open the environment.
        if self._temporal_segments:
            self._temporal_segments = {}
            self.btn_goto_segment.setEnabled(False)
            self.btn_insp_capture.setEnabled(False)
        self._refresh_status_panel()

    def on_open_env_clicked(self):
        day = self._selected_day()
        seg = self._selected_segment()
        self.btn_open_env.setEnabled(False)
        self.btn_open_env.setText("⏳ Materialising…")
        self._refresh_status_panel(f"⏳ Materialising {seg} Day {day:02d} in Unreal Engine…")
        QtWidgets.QApplication.processEvents()

        res = self.ipc.send_command(
            {"action": "materialize_segment_day", "segment_id": seg, "day": day},
            timeout=30.0,
        )

        self.btn_open_env.setEnabled(True)
        self.btn_open_env.setText("🛠  Materialise & Inspect")

        if res.get("status") == "ok":
            state_name = res.get("deterioration_state", "Materialised")
            sev = res.get("severity_value", 0.0)
            n_defects = res.get("total_defects", 0)
            self._temporal_segments = {seg: res}
            self.btn_goto_segment.setEnabled(True)
            self.btn_insp_capture.setEnabled(True)
            self._refresh_status_panel(
                f"{seg} | Day {day:02d}\nState: {state_name}\nSeverity: {sev:.2f} · {n_defects} defect(s)"
            )
            self.status.showMessage(
                f"✓ Materialised {seg} Day {day:02d} ({state_name}) — camera positioned top-down."
            )
        else:
            err = res.get("error") or res.get("message", "Unknown error")
            self._refresh_status_panel(f"✗ Error: {err}")
            self.status.showMessage(f"✗ Failed to materialise {seg} Day {day:02d}: {err}")

    def on_goto_segment_clicked(self):
        seg = self._selected_segment()
        day = self._selected_day()
        self.status.showMessage(f"Moving camera to {seg} (Day {day:02d})…")
        QtWidgets.QApplication.processEvents()

        res = self.ipc.send_command(
            {"action": "temporal_goto_segment", "segment_id": seg},
            timeout=10.0,
        )
        if res.get("status") == "ok":
            self._refresh_status_panel(f"📍 Camera at {seg}")
            self.status.showMessage(
                f"📍 Viewport moved to {seg} — downward-facing inspection pose."
            )
        else:
            err = res.get("error") or res.get("message", "Unknown error")
            self.status.showMessage(f"✗ Goto segment failed: {err}")

    def on_insp_capture_clicked(self):
        day = self._selected_day()
        seg = self._selected_segment()
        metadata_snapshot = self._safe_inspection_metadata_snapshot()

        self.btn_insp_capture.setEnabled(False)
        self.btn_insp_capture.setText("⏳ Capturing…")
        self.lbl_insp_saved.setText("")
        self._refresh_status_panel("⏳ SceneCapture2D rendering…")
        QtWidgets.QApplication.processEvents()

        res = self.ipc.send_command(
            {"action": "inspection_capture", "day": day, "segment_id": seg},
            timeout=45.0,
        )

        self.btn_insp_capture.setEnabled(True)
        self.btn_insp_capture.setText("📸 Capture Inspection  [C]")

        if res.get("status") in ("ok", "warning"):
            path = res.get("path", "")
            validated = res.get("validated", False)
            reason = res.get("validation_reason") or ""
            size_kb = res.get("file_size_bytes", 0) // 1024

            if validated:
                saved_rel = Path(path).relative_to(WORKSPACE_ROOT) if path else path
                self.lbl_insp_saved.setText(
                    f"✓ Saved ({size_kb} KB): {saved_rel}"
                )
                self._refresh_status_panel(f"✓ Capture saved ({size_kb} KB)")
                self.status.showMessage(
                    f"📸 Inspection captured: {Path(path).name}  ({size_kb} KB)"
                )
            else:
                self.lbl_insp_saved.setText(
                    f"⚠ Capture saved but validation failed: {reason}"
                )
                self._refresh_status_panel(f"⚠ Validation: {reason}")
                self.status.showMessage(f"⚠ Capture may be blank: {reason}")
        else:
            err = res.get("error") or res.get("message", "Unknown IPC error")
            self.lbl_insp_saved.setText(f"✗ Capture failed: {err}")
            self._refresh_status_panel(f"✗ Capture error: {err}")
            self.status.showMessage(f"✗ Inspection capture failed: {err}")

        if metadata_snapshot is not None:
            self._log_inspection_metadata_passively(metadata_snapshot, res)

    # ------------------------------------------------------------------
    # Existing Studio Slots (unchanged)
    # ------------------------------------------------------------------

    def on_density_changed(self, val):
        density = val / 10.0
        self.lbl_density.setText(f"{density:.1f} / 100m²")

    def on_lighting_changed(self):
        preset = self.combo_light.currentText()
        res = self.ipc.send_command({"action": "lighting", "preset": preset})
        if res.get("status") == "ok":
            self.status.showMessage(f"Lighting updated: {preset}")

    def on_teleport_clicked(self):
        vp_name = self.combo_viewpoint.currentText()
        res = self.ipc.send_command({"action": "teleport", "viewpoint": vp_name})
        if res.get("status") == "ok":
            self.status.showMessage(f"Camera moved to: {vp_name}")

    def on_capture_clicked(self):
        self.status.showMessage("Capturing high-resolution drone photo…")
        QtWidgets.QApplication.processEvents()
        res = self.ipc.send_command({"action": "capture"})
        if res.get("status") == "ok":
            path = res.get("path", "")
            fname = os.path.basename(path) if path else f"drone_capture_{int(time.time())}.png"
            self.status.showMessage(f"📸 Captured photo: {fname} saved to env/output/captures/")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"drone_capture_{ts}.png"
            self.status.showMessage(f"📸 Captured photo: {fname} saved to env/output/captures/")

    def on_generate_clicked(self):
        health = self.combo_health.currentText()
        light = self.combo_light.currentText()
        sizing = self.combo_sizing.currentText()
        density = self.slider_density.value() / 10.0
        water = self.combo_water.currentText()

        self.status.showMessage("Generating 3D world in Unreal Engine…")
        QtWidgets.QApplication.processEvents()

        res = self.ipc.send_command({
            "action": "generate",
            "road_health": health,
            "lighting": light,
            "sizing": sizing,
            "density": density,
            "water": water
        })

        if res.get("status") == "ok":
            count = res.get("total_defects", "varied")
            self.status.showMessage(f"✓ World Generated: {count} defects spawned. Condition: {health}")
        else:
            self.status.showMessage(f"Generation command dispatched: {health}, {light}")

    # ------------------------------------------------------------------
    # Key Events
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key.Key_C:
            # C key always triggers the inspection capture if a scene is loaded,
            # otherwise fall back to the existing drone capture.
            if self._temporal_segments and self.btn_insp_capture.isEnabled():
                self.on_insp_capture_clicked()
            else:
                self.on_capture_clicked()
            event.accept()
        else:
            super().keyPressEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = RoadSentinelStudioWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
