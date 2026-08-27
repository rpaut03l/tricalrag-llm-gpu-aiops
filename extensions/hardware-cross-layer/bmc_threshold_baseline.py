"""
WHY this baseline: threshold-based alerting (if temp > X, alert) is the
industry-standard approach every BMC/iDRAC/iLO ships with out of the box.
It's the real-world incumbent this project's LLM-based cross-layer
attribution has to beat or match to make a meaningful claim. Without this,
"LLM does cross-layer RCA" is an unfalsifiable claim — this baseline makes
it testable.

Thresholds below are illustrative, modeled on typical datacenter GPU
server BMC defaults — cite your specific hardware vendor's actual
thresholds in the paper if available.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

THRESHOLDS = {
    "gpu_temp_c_max": 83.0,
    "fan_rpm_min": 500,
    "ecc_uncorrected_max": 0,       # any uncorrected ECC error is critical
    "psu_voltage_band": (11.4, 12.6),  # +/- 5% of nominal 12V
}


def threshold_classify(hw_window):
    """Returns predicted fault type using simple static thresholds — no learning."""
    max_temp = max(hw_window["gpu_temp_c"])
    min_fan = min(hw_window["fan_rpm"])
    max_uncorrected = max(hw_window["ecc_uncorrected_total"])
    voltages = hw_window["psu_voltage_12v"]
    voltage_out_of_band = any(
        v < THRESHOLDS["psu_voltage_band"][0] or v > THRESHOLDS["psu_voltage_band"][1]
        for v in voltages
    )

    if max_uncorrected > THRESHOLDS["ecc_uncorrected_max"]:
        return "ECC_ERROR_BURST"
    if min_fan < THRESHOLDS["fan_rpm_min"]:
        return "FAN_FAILURE"
    if max_temp > THRESHOLDS["gpu_temp_c_max"]:
        return "THERMAL_THROTTLE"
    if voltage_out_of_band:
        return "PSU_INSTABILITY"
    return "NONE"


def main():
    with open("../../benchmark/data/cross_layer_incidents.jsonl") as f:
        incidents = [json.loads(line) for line in f]

    preds, gt = [], []
    for inc in incidents:
        pred = threshold_classify(inc["hardware_telemetry"])
        preds.append(pred)
        gt.append(inc["hardware_fault_label"])

    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average="macro", zero_division=0)
    print(f"Threshold-based BMC baseline — Accuracy: {acc:.3f}, Macro-F1: {f1:.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/bmc_threshold_baseline.json", "w") as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "thresholds": THRESHOLDS}, f, indent=2)
    print("Saved -> results/bmc_threshold_baseline.json")


if __name__ == "__main__":
    main()
