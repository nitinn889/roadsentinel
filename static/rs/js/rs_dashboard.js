// RoadSentinel Dashboard - Interactive Client Controller

document.addEventListener("DOMContentLoaded", () => {
  initLiveTelemetry();
  initControlForm();
});

// Toast Notification
function showToast(message, isError = false) {
  let toast = document.getElementById("rs-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "rs-toast";
    toast.className = "rs-toast";
    document.body.appendChild(toast);
  }
  toast.innerText = message;
  toast.className = isError ? "rs-toast show error" : "rs-toast show";
  setTimeout(() => {
    toast.className = "rs-toast";
  }, 3500);
}

// Telemetry Poller
function initLiveTelemetry() {
  const statusBadge = document.getElementById("val-status-pill");
  const weatherBadge = document.getElementById("val-weather-badge");
  const overlayWeather = document.getElementById("overlay-weather-badge");
  const overlayMode = document.getElementById("overlay-mode-badge");
  const valAltitude = document.getElementById("val-altitude");
  const valSpeed = document.getElementById("val-speed");
  const valDefects = document.getElementById("val-defects-count");
  const valFrames = document.getElementById("val-frame-count");
  const valDisk = document.getElementById("val-disk-free");

  function pollStatus() {
    fetch("/rs/status")
      .then(res => res.json())
      .then(data => {
        if (weatherBadge && data.active_preset) {
          weatherBadge.innerText = data.active_preset;
        }
        if (overlayWeather && data.active_preset) {
          overlayWeather.innerText = "Weather: " + data.active_preset;
        }
        if (overlayMode && data.flight_mode) {
          overlayMode.innerText = "Mode: " + data.flight_mode.toUpperCase();
        }
        if (valAltitude && data.drone_position) {
          valAltitude.innerText = data.drone_position.z.toFixed(1);
        }
        if (valFrames) {
          valFrames.innerText = data.frame_count || 0;
        }
        if (valDisk && data.disk_usage) {
          valDisk.innerText = data.disk_usage.free_gb.toFixed(2);
          if (data.disk_usage.warning) {
            valDisk.style.color = "var(--accent-red)";
            showToast("Disk space warning: " + data.disk_usage.warning_msg, true);
          }
        }
      })
      .catch(err => console.debug("Status poll error:", err));
  }

  // Poll status every 1.5 seconds
  setInterval(pollStatus, 1500);
  pollStatus();
}

// Preset visual thumbnails
const PRESET_ICONS = {
  "ClearNoon": "☀️",
  "OvercastDay": "☁️",
  "HeavyRain": "🌧️",
  "LightDrizzle": "🌦️",
  "DuskGoldenHour": "🌅",
  "NightClear": "🌙",
  "NightFoggy": "🌫️",
  "NightRain": "⛈️"
};

// Parameter Form Controller
function initControlForm() {
  const form = document.getElementById("rs-control-form");
  if (!form) return;

  const densitySlider = document.getElementById("input-defect-density");
  const densityVal = document.getElementById("val-defect-density");
  const altitudeSlider = document.getElementById("input-altitude");
  const altitudeVal = document.getElementById("val-altitude-display");
  const weatherSelect = document.getElementById("input-weather-preset");
  const presetThumbnail = document.getElementById("preset-thumbnail-icon");
  const presetDesc = document.getElementById("preset-thumbnail-desc");
  const captureBtn = document.getElementById("btn-toggle-capture");

  // Slider feedback
  if (densitySlider && densityVal) {
    densitySlider.addEventListener("input", (e) => {
      densityVal.innerText = parseFloat(e.target.value).toFixed(1);
    });
  }

  if (altitudeSlider && altitudeVal) {
    altitudeSlider.addEventListener("input", (e) => {
      altitudeVal.innerText = parseFloat(e.target.value).toFixed(0) + " m";
    });
  }

  // Weather preview update
  if (weatherSelect && presetThumbnail) {
    weatherSelect.addEventListener("change", (e) => {
      const selected = e.target.value;
      presetThumbnail.innerText = PRESET_ICONS[selected] || "🌤️";
      if (presetDesc) {
        presetDesc.innerText = e.target.selectedOptions[0].getAttribute("data-desc") || selected;
      }
    });
  }

  // Form Submit via fetch (no page reload)
  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const selectedDefectTypes = Array.from(
      form.querySelectorAll("input[name='defect_types']:checked")
    ).map(cb => cb.value);

    const flightModeRadio = form.querySelector("input[name='flight_mode']:checked");

    const payload = {
      weather_preset: weatherSelect ? weatherSelect.value : "ClearNoon",
      defect_density: densitySlider ? parseFloat(densitySlider.value) : 3.0,
      defect_types: selectedDefectTypes,
      water_ior: parseFloat(document.getElementById("input-water-ior")?.value || 1.333),
      altitude_m: altitudeSlider ? parseFloat(altitudeSlider.value) : 25.0,
      flight_mode: flightModeRadio ? flightModeRadio.value : "hover"
    };

    fetch("/rs/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) throw new Error("HTTP error " + res.status);
        return res.json();
      })
      .then(data => {
        showToast("Simulation parameters applied successfully!");
      })
      .catch(err => {
        showToast("Error updating parameters: " + err.message, true);
      });
  });

  // Capture Toggle Button
  if (captureBtn) {
    let capturing = false;
    captureBtn.addEventListener("click", () => {
      capturing = !capturing;
      if (capturing) {
        captureBtn.innerText = "⏹ Stop Capture";
        captureBtn.className = "rs-btn rs-btn-danger";
      } else {
        captureBtn.innerText = "⏺ Start Capture";
        captureBtn.className = "rs-btn rs-btn-primary";
      }

      fetch("/rs/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_capturing: capturing })
      })
        .then(res => res.json())
        .then(data => {
          showToast(capturing ? "Sensor capture active." : "Sensor capture paused.");
        });
    });
  }
}
