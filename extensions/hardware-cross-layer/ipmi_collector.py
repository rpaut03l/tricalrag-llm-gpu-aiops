"""
Collects hardware telemetry via IPMI/BMC from the host machine.

WHY: application-level log anomalies (what LogSentinel-RAG originally
benchmarks) can't distinguish "the software has a bug" from "the GPU is
thermal-throttling" or "a PSU is failing" — those signals only exist at
the hardware/BMC layer. This module pulls that layer in.

IMPORTANT HONESTY NOTE: not every machine has a BMC. Workstation-class
motherboards (common in single-GPU research rigs) often do NOT expose
IPMI — that's typically a SERVER-motherboard feature (Supermicro, Dell
iDRAC, HPE iLO). This script detects that gracefully and reports it
rather than failing silently or fabricating data.

Requires `ipmitool` installed:
    sudo apt install ipmitool

Usage:
    python ipmi_collector.py --check        # just check if BMC is present
    python ipmi_collector.py --collect      # pull one telemetry snapshot
    python ipmi_collector.py --sel          # pull System Event Log (hardware fault history)
"""

import subprocess
import json
import re
import argparse
from datetime import datetime, timezone
from pathlib import Path


def check_bmc_present():
    """Returns True if a local BMC responds to ipmitool, False otherwise."""
    try:
        result = subprocess.run(
            ["ipmitool", "sensor", "list"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except FileNotFoundError:
        print("ipmitool not installed. Run: sudo apt install ipmitool")
        return False
    except subprocess.TimeoutExpired:
        print("ipmitool timed out — likely no BMC present on this hardware.")
        return False
    except Exception as e:
        print(f"BMC check failed: {e}")
        return False


def parse_sensor_line(line):
    """
    ipmitool sensor output format (pipe-separated):
    Sensor Name | Value | Units | Status | Lower NR | Lower CR | Lower NC | Upper NC | Upper CR | Upper NR
    """
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 4:
        return None
    name, value, units, status = parts[0], parts[1], parts[2], parts[3]
    try:
        value = float(value)
    except ValueError:
        value = None
    return {"sensor": name, "value": value, "units": units, "status": status}


def collect_sensor_snapshot():
    """Pull one point-in-time reading of all BMC sensors (temps, fans, power, voltages)."""
    try:
        result = subprocess.run(["ipmitool", "sensor", "list"],
                                 capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        readings = [parse_sensor_line(l) for l in result.stdout.strip().split("\n")]
        readings = [r for r in readings if r is not None]
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensors": readings,
        }
    except Exception as e:
        print(f"Sensor collection failed: {e}")
        return None


def collect_sel_log():
    """
    Pull the System Event Log — the BMC's own hardware fault history
    (PSU failures, ECC errors, thermal trips, fan failures, etc.)
    This is the hardware-layer equivalent of an application log file.
    """
    try:
        result = subprocess.run(["ipmitool", "sel", "list"],
                                 capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return []
        entries = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split("|")]
            entries.append({"raw": line.strip(), "fields": parts})
        return entries
    except Exception as e:
        print(f"SEL collection failed: {e}")
        return []


def collect_nvidia_smi_correlation():
    """
    Supplement BMC data with nvidia-smi GPU-level telemetry (temp, power draw,
    ECC error counts) — this works even WITHOUT a server-class BMC, since
    nvidia-smi talks to the GPU directly regardless of motherboard type.
    This is the fallback hardware signal for workstation-class machines.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=timestamp,temperature.gpu,power.draw,fan.speed,"
             "ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,"
             "clocks_throttle_reasons.hw_slowdown",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
        return {"raw": result.stdout.strip()}
    except Exception as e:
        print(f"nvidia-smi hardware telemetry failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--sel", action="store_true")
    ap.add_argument("--out", default="../../benchmark/data/hardware_telemetry.jsonl")
    args = ap.parse_args()

    bmc_present = check_bmc_present()
    print(f"BMC/IPMI present: {bmc_present}")

    if not bmc_present:
        print("\nNo server-class BMC detected on this machine. This is EXPECTED "
              "for most workstation motherboards (BMC/IPMI is typically a "
              "server-hardware feature — Supermicro, Dell iDRAC, HPE iLO, etc.).\n"
              "Falling back to nvidia-smi GPU-level telemetry as the hardware signal.")
        gpu_data = collect_nvidia_smi_correlation()
        print(json.dumps(gpu_data, indent=2))
        return

    if args.check:
        return

    if args.collect:
        snapshot = collect_sensor_snapshot()
        gpu_data = collect_nvidia_smi_correlation()
        combined = {"bmc_sensors": snapshot, "gpu_telemetry": gpu_data}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(combined) + "\n")
        print(f"Snapshot appended to {args.out}")

    if args.sel:
        sel = collect_sel_log()
        print(f"Found {len(sel)} SEL entries")
        for entry in sel[:10]:
            print(entry["raw"])


if __name__ == "__main__":
    main()
