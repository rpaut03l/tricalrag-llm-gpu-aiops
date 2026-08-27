"""
Prompt templates for the PRELIMINARY, SYNTHETIC extensions (cross-layer
hardware attribution and full-stack trust/provisioning/orchestration
attribution). See extensions/README.md for scope — these are not part
of the core benchmark's validated results.
"""

CROSS_LAYER_TEMPLATE = """You are an SRE assistant diagnosing an incident using BOTH application logs
AND hardware telemetry from the underlying server's BMC/GPU sensors.

--- APPLICATION LOG WINDOW ---
{log_window}
--- END APPLICATION LOGS ---

--- HARDWARE TELEMETRY (same time window) ---
GPU Temperature (C): {gpu_temp}
Power Draw (W): {power_draw}
Fan RPM: {fan_rpm}
ECC Corrected Errors: {ecc_corrected}
ECC Uncorrected Errors: {ecc_uncorrected}
PSU 12V Rail (V): {psu_voltage}
--- END HARDWARE TELEMETRY ---

Determine whether this incident's root cause is at the SOFTWARE layer, the
HARDWARE layer, BOTH, or NONE (no real incident). Consider: thermal throttling
shows as a temperature ramp with falling power draw; ECC bursts show as sudden
error-count spikes; fan failure shows as RPM collapse with rising temperature;
PSU instability shows as voltage oscillation outside a tight band around 12V.

Respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "attribution": "SOFTWARE" | "HARDWARE" | "BOTH" | "NONE",
  "hardware_fault_type": "THERMAL_THROTTLE" | "ECC_ERROR_BURST" | "PSU_INSTABILITY" | "FAN_FAILURE" | "NONE",
  "confidence": "low" | "medium" | "high",
  "root_cause": "one sentence explanation spanning both layers if relevant",
  "remediation": "one sentence suggested fix"
}}
"""

def build_cross_layer_prompt(log_window: str, hw_telemetry: dict) -> str:
    return CROSS_LAYER_TEMPLATE.format(
        log_window=log_window,
        gpu_temp=hw_telemetry["gpu_temp_c"],
        power_draw=hw_telemetry["power_draw_w"],
        fan_rpm=hw_telemetry["fan_rpm"],
        ecc_corrected=hw_telemetry["ecc_corrected_total"],
        ecc_uncorrected=hw_telemetry["ecc_uncorrected_total"],
        psu_voltage=hw_telemetry["psu_voltage_12v"],
    )


FULLSTACK_TEMPLATE = """You are an SRE assistant diagnosing an incident on a bare-metal AI
infrastructure node. You have signals from THREE layers below the
application. Determine which layer(s), if any, are the true root cause.

--- TRUST/BOOT LAYER (Secure Boot, TPM attestation, cert chain) ---
{trust_chain_log}

--- PROVISIONING LAYER (PXE/DHCP/TFTP netboot, cloud-init, BMC) ---
{provisioning_log}

--- ORCHESTRATION LAYER (etcd, node readiness, dataplane/CNI, certs) ---
{orchestration_log}

A node that never joined the cluster could be explained by a provisioning
failure (never booted), a trust-chain failure (boot was blocked by security
verification), an orchestration failure (booted fine but couldn't join the
cluster), or some combination. Evaluate each layer's evidence independently.

Respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "faulty_layers": ["TRUST_CHAIN" | "PROVISIONING" | "ORCHESTRATION" | "NONE", ...],
  "primary_root_cause": "one sentence identifying the earliest/most fundamental fault in the chain",
  "remediation": "one sentence suggested fix, addressing the earliest layer first"
}}
"""

def build_fullstack_prompt(trust_chain_log: str, provisioning_log: str, orchestration_log: str) -> str:
    return FULLSTACK_TEMPLATE.format(
        trust_chain_log=trust_chain_log,
        provisioning_log=provisioning_log,
        orchestration_log=orchestration_log,
    )
