"""
Extends the cross-layer RCA task with TWO more layers beyond hardware:

  LAYER: TRUST_CHAIN   — boot-time integrity (Secure Boot, TPM attestation)
  LAYER: PROVISIONING  — bare-metal bring-up (PXE/DHCP/TFTP, cloud-init, BMC reachability)
  LAYER: ORCHESTRATION — cluster-level (etcd quorum, node readiness, dataplane config, cert expiry)

WHY these layers, grounded in real operational practice (genericized —
no vendor/company names):
- Bare-metal fleets boot via PXE (DHCP DISCOVER/OFFER/REQUEST/ACK, then
  TFTP-fetched bootloader+kernel) BEFORE any OS-level logging exists.
  A boot-time failure produces NO application logs at all — which is
  itself a diagnostic signal our application-log-only baseline cannot see.
- Trusted-boot chains (TPM measured boot + Secure Boot signature
  verification + remote attestation) are how modern fleets prove a node
  wasn't tampered with. A node failing attestation looks, from the
  orchestration layer, identical to "node never came up" — but the true
  root cause and remediation are completely different (compromised
  firmware vs. a simple provisioning misconfiguration).
- Cluster orchestration failures (etcd quorum loss, node NotReady,
  dataplane/CNI misconfig, cert expiry) are the most common real-world
  bare-metal Kubernetes incident categories reported by platform teams.

As with hardware fault injection, this uses CONTROLLED SYNTHETIC GENERATION
of realistic incident narratives — there is no public dataset of real
boot/provisioning/orchestration incidents paired with labels. Report this
transparently in the paper's methodology.
"""

import json
import random
from pathlib import Path

TRUST_CHAIN_FAULTS = [
    "SECURE_BOOT_VIOLATION",     # bootloader/kernel signature check failed
    "TPM_ATTESTATION_MISMATCH",  # measured boot state doesn't match known-good PCR values
    "CERT_CHAIN_INVALID",        # TLS cert chain broken/untrusted at node join time
    "NONE",
]

PROVISIONING_FAULTS = [
    "BMC_UNREACHABLE",           # out-of-band management network down
    "PXE_DHCP_TIMEOUT",          # no DHCP OFFER received during netboot
    "TFTP_FETCH_FAILURE",        # bootloader/kernel fetch failed after DHCP succeeded
    "CLOUD_INIT_FAILURE",        # post-boot provisioning script failed
    "NONE",
]

ORCHESTRATION_FAULTS = [
    "ETCD_QUORUM_LOSS",          # control-plane consensus broken
    "NODE_NOT_READY",            # kubelet/node registration failure
    "DATAPLANE_MISCONFIG",       # CNI/SR-IOV/dataplane interface misconfigured
    "CERT_EXPIRY",               # cluster-internal TLS cert expired
    "NONE",
]

# --- Narrative templates: realistic synthetic log/event lines per fault type ---

TRUST_CHAIN_TEMPLATES = {
    "SECURE_BOOT_VIOLATION": [
        "UEFI: Secure Boot violation — bootloader signature verification failed",
        "shim: Invalid signature detected, refusing to load unsigned kernel image",
        "UEFI: boot halted — untrusted binary in boot chain",
    ],
    "TPM_ATTESTATION_MISMATCH": [
        "tpm2: PCR measurement mismatch — expected {expected}, got {actual}",
        "attestation-agent: remote attestation FAILED — measured boot state does not match known-good baseline",
        "tpm2: sealed key unseal failed — boot state verification error",
    ],
    "CERT_CHAIN_INVALID": [
        "tls: certificate signed by unknown authority during node join",
        "x509: certificate has expired or is not yet valid",
        "handshake failure: unable to verify certificate chain",
    ],
    "NONE": [
        "UEFI: Secure Boot verification passed, all signatures valid",
        "tpm2: attestation successful — boot state matches known-good baseline",
    ],
}

PROVISIONING_TEMPLATES = {
    "BMC_UNREACHABLE": [
        "ipmitool: Unable to establish LAN session — BMC not responding",
        "provisioning-agent: out-of-band management interface unreachable, retrying (3/5)",
        "network: no route to BMC management VLAN",
    ],
    "PXE_DHCP_TIMEOUT": [
        "dhclient: No DHCPOFFER received after 60s, PXE boot aborted",
        "PXE-E51: No DHCP or proxyDHCP offers were received",
        "netboot: DHCP DISCOVER sent, no response on provisioning VLAN",
    ],
    "TFTP_FETCH_FAILURE": [
        "PXE-E32: TFTP open timeout fetching bootloader",
        "tftp: connection refused while fetching pxelinux.0",
        "netboot: kernel/initrd fetch failed after successful DHCP handshake",
    ],
    "CLOUD_INIT_FAILURE": [
        "cloud-init: stage 'modules:final' failed, see /var/log/cloud-init.log",
        "cloud-init: datasource not found after 120s, falling back to None",
        "cloud-init: user-data script exited with non-zero status",
    ],
    "NONE": [
        "provisioning-agent: node provisioned successfully, handing off to OS install",
        "cloud-init: finished successfully in 48.2s",
    ],
}

ORCHESTRATION_TEMPLATES = {
    "ETCD_QUORUM_LOSS": [
        "etcdserver: cluster lost quorum, no leader elected",
        "etcd: request timed out, possibly due to lost quorum",
        "etcdserver: rejecting write request — no leader",
    ],
    "NODE_NOT_READY": [
        "kubelet: node status NotReady — PLEG is not healthy",
        "kube-controller-manager: node not responding, marking NotReady",
        "kubelet: failed to register node — connection refused",
    ],
    "DATAPLANE_MISCONFIG": [
        "cni: failed to set up pod network — SR-IOV VF allocation failed",
        "multus: no available network attachment for requested interface",
        "cni: dataplane interface down, packet forwarding disabled",
    ],
    "CERT_EXPIRY": [
        "kube-apiserver: client certificate has expired",
        "x509: certificate has expired, cluster-internal TLS handshake failed",
        "kubelet: unable to rotate certificate — CSR approval timeout",
    ],
    "NONE": [
        "etcdserver: leader elected, cluster healthy, quorum stable",
        "kubelet: node status Ready",
    ],
}


def _pick_narrative(fault_templates, fault_type, rng):
    lines = fault_templates[fault_type]
    n = rng.randint(1, min(3, len(lines)))
    return "\n".join(rng.sample(lines, n))


def generate_layered_incident(incident_id, rng, fault_probability=0.35):
    """
    Generates one synthetic full-stack incident with independently-decided
    fault states at each of the three new layers (trust, provisioning,
    orchestration). Layers are DECOUPLED — a trust-chain fault does not
    imply a provisioning fault — so the model must reason about each
    layer's evidence independently, then determine overall attribution.
    """
    trust_fault = rng.choice(TRUST_CHAIN_FAULTS[:-1]) if rng.random() < fault_probability else "NONE"
    provisioning_fault = rng.choice(PROVISIONING_FAULTS[:-1]) if rng.random() < fault_probability else "NONE"
    orchestration_fault = rng.choice(ORCHESTRATION_FAULTS[:-1]) if rng.random() < fault_probability else "NONE"

    faults_present = [f for f in [trust_fault, provisioning_fault, orchestration_fault] if f != "NONE"]

    return {
        "id": incident_id,
        "trust_chain_log": _pick_narrative(TRUST_CHAIN_TEMPLATES, trust_fault, rng),
        "trust_chain_fault": trust_fault,
        "provisioning_log": _pick_narrative(PROVISIONING_TEMPLATES, provisioning_fault, rng),
        "provisioning_fault": provisioning_fault,
        "orchestration_log": _pick_narrative(ORCHESTRATION_TEMPLATES, orchestration_fault, rng),
        "orchestration_fault": orchestration_fault,
        "any_fault_present": len(faults_present) > 0,
        "true_faulty_layers": faults_present if faults_present else ["NONE"],
    }


def build_fullstack_dataset(n=150, seed=42):
    rng = random.Random(seed)
    return [generate_layered_incident(i, rng) for i in range(n)]


if __name__ == "__main__":
    dataset = build_fullstack_dataset()
    with open("../../benchmark/data/fullstack_incidents.jsonl", "w") as f:
        for inc in dataset:
            f.write(json.dumps(inc) + "\n")

    from collections import Counter
    layer_counts = Counter()
    for inc in dataset:
        for layer in inc["true_faulty_layers"]:
            layer_counts[layer] += 1
    print(f"Wrote {len(dataset)} full-stack incidents to data/fullstack_incidents.jsonl")
    print(f"Fault distribution across layers: {dict(layer_counts)}")
    print("\nNOTE: these are SYNTHETICALLY GENERATED narrative incidents modeled on "
          "documented, generic industry failure patterns (PXE/DHCP/TFTP boot chain, "
          "TPM/Secure Boot attestation, etcd/Kubernetes orchestration). No real "
          "infrastructure, company, or vendor-specific data is used or referenced.")
