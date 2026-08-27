"""
Builds the CROSS-LAYER incident dataset: pairs each software-log incident
(from BGL/HDFS/Thunderbird/OpenStack) with a hardware telemetry window that
is EITHER (a) normal baseline readings, or (b) a synthetically injected
fault signature representing a specific known hardware failure mode.

WHY SYNTHETIC INJECTION, STATED HONESTLY:
No public dataset pairs real BMC/IPMI telemetry with real software incidents
at scale — this combination doesn't exist in LogHub or anywhere else public.
Rather than fabricate a false claim of "real hardware failure data," we use
CONTROLLED SYNTHETIC INJECTION: realistic fault signatures modeled on
published hardware failure characteristics (thermal throttling curves, ECC
error burst patterns, PSU ripple/voltage sag signatures), injected at known
labeled points. This is a standard, legitimate methodology in fault-injection
and chaos-engineering research — but it MUST be reported as synthetic in the
paper's methodology section, not conflated with real-world hardware failure
data. This script's output should always be described as such.

Fault modes modeled (each grounded in a real, documented failure signature):
  1. THERMAL_THROTTLE   - GPU temp ramps toward/above throttle threshold (~83-90C for RTX PRO 6000 class),
                           power draw drops as the card self-limits clocks
  2. ECC_ERROR_BURST    - sudden spike in corrected/uncorrected ECC error counts (memory degradation)
  3. PSU_INSTABILITY    - voltage sensor readings oscillate outside nominal band
  4. FAN_FAILURE        - fan RPM drops toward zero while temp keeps climbing (no compensating airflow)
  5. NONE (baseline)    - normal, stable telemetry with no injected fault
"""

import json
import random
import numpy as np
from pathlib import Path

FAULT_MODES = ["NONE", "THERMAL_THROTTLE", "ECC_ERROR_BURST", "PSU_INSTABILITY", "FAN_FAILURE"]
WINDOW_LEN = 10          # telemetry samples per window (e.g. 10 seconds of readings)
SEED = 42


def generate_baseline_window(rng):
    """Normal, stable hardware telemetry — no fault."""
    return {
        "gpu_temp_c": list(np.round(rng.normal(55, 2, WINDOW_LEN), 1)),
        "power_draw_w": list(np.round(rng.normal(180, 8, WINDOW_LEN), 1)),
        "fan_rpm": list(np.round(rng.normal(2200, 100, WINDOW_LEN), 0)),
        "ecc_corrected_total": [0] * WINDOW_LEN,
        "ecc_uncorrected_total": [0] * WINDOW_LEN,
        "psu_voltage_12v": list(np.round(rng.normal(12.0, 0.05, WINDOW_LEN), 3)),
    }


def inject_fault(baseline, fault_mode, rng):
    """Overlay a realistic fault signature onto an otherwise-normal window."""
    w = {k: list(v) for k, v in baseline.items()}

    if fault_mode == "THERMAL_THROTTLE":
        # Temp ramps up across the window, power draw self-limits as it nears threshold
        ramp = np.linspace(0, 30, WINDOW_LEN)
        w["gpu_temp_c"] = list(np.round(np.array(w["gpu_temp_c"]) + ramp, 1))
        w["power_draw_w"] = list(np.round(
            np.array(w["power_draw_w"]) * np.linspace(1.0, 0.7, WINDOW_LEN), 1))

    elif fault_mode == "ECC_ERROR_BURST":
        burst_point = WINDOW_LEN // 2
        corrected = [0] * burst_point + list(rng.integers(5, 40, WINDOW_LEN - burst_point))
        uncorrected = [0] * (WINDOW_LEN - 2) + [0, rng.integers(1, 3)]
        w["ecc_corrected_total"] = corrected
        w["ecc_uncorrected_total"] = uncorrected

    elif fault_mode == "PSU_INSTABILITY":
        noise = rng.normal(0, 0.4, WINDOW_LEN)  # abnormal voltage oscillation
        w["psu_voltage_12v"] = list(np.round(np.array(w["psu_voltage_12v"]) + noise, 3))

    elif fault_mode == "FAN_FAILURE":
        decay = np.linspace(1.0, 0.05, WINDOW_LEN)
        w["fan_rpm"] = list(np.round(np.array(w["fan_rpm"]) * decay, 0))
        # temp climbs because airflow is gone
        ramp = np.linspace(0, 20, WINDOW_LEN)
        w["gpu_temp_c"] = list(np.round(np.array(w["gpu_temp_c"]) + ramp, 1))

    return w


def build_cross_layer_dataset(software_incidents, seed=SEED, hardware_fault_ratio=0.3):
    """
    Pairs each software incident with a hardware telemetry window.
    ~30% of incidents get a genuine hardware fault injected (by default);
    the rest get normal baseline hardware telemetry, regardless of whether
    the software layer shows an anomaly. This deliberately DECOUPLES the two
    layers so the LLM must actually reason about attribution rather than
    pattern-match "software anomaly implies hardware fault."
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    combined = []
    for inc in software_incidents:
        baseline = generate_baseline_window(rng)
        has_hw_fault = py_rng.random() < hardware_fault_ratio
        fault_mode = py_rng.choice(FAULT_MODES[1:]) if has_hw_fault else "NONE"
        hw_window = inject_fault(baseline, fault_mode, rng) if has_hw_fault else baseline

        combined.append({
            **inc,
            "hardware_telemetry": hw_window,
            "hardware_fault_label": fault_mode,
            "hardware_fault_injected": has_hw_fault,
            # Ground truth for the cross-layer attribution task:
            # what's the TRUE root cause layer for this combined incident?
            "true_attribution": (
                "HARDWARE" if has_hw_fault and inc["is_anomaly"] else
                "SOFTWARE" if inc["is_anomaly"] and not has_hw_fault else
                "BOTH" if has_hw_fault and inc["is_anomaly"] else
                "NONE"
            ),
        })
    return combined


if __name__ == "__main__":
    with open("../../benchmark/data/incidents.jsonl") as f:
        software_incidents = [json.loads(line) for line in f]

    combined = build_cross_layer_dataset(software_incidents)

    with open("../../benchmark/data/cross_layer_incidents.jsonl", "w") as f:
        for inc in combined:
            f.write(json.dumps(inc) + "\n")

    from collections import Counter
    attribution_counts = Counter(i["true_attribution"] for i in combined)
    fault_counts = Counter(i["hardware_fault_label"] for i in combined)
    print(f"Wrote {len(combined)} cross-layer incidents to data/cross_layer_incidents.jsonl")
    print(f"Attribution breakdown: {dict(attribution_counts)}")
    print(f"Fault mode breakdown: {dict(fault_counts)}")
    print("\nNOTE: hardware fault signatures are SYNTHETICALLY INJECTED, modeled on "
          "documented failure characteristics. Report this clearly in the paper's "
          "methodology — do not present as real hardware failure telemetry.")
