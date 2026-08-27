# Extensions — Preliminary / Synthetic Feasibility Work

**These are not part of the core benchmark's validated results.** Everything
in this folder uses synthetically generated or synthetically injected data,
because no public dataset exists pairing real BMC/IPMI hardware telemetry,
boot-trust events, or bare-metal provisioning logs with labeled software
incidents at scale.

## What's here

- **`hardware-cross-layer/`** — software-log + hardware-telemetry (GPU/BMC)
  joint attribution, with synthetically injected fault signatures (thermal,
  ECC, PSU, fan) modeled on documented failure characteristics.
- **`fullstack-trust-provisioning/`** — synthetic narrative incidents at the
  boot-trust (Secure Boot/TPM), provisioning (PXE/DHCP/TFTP/cloud-init), and
  orchestration (etcd/Kubernetes) layers.

## Why these are kept separate from the core benchmark

The [core benchmark](../README.md) makes a narrow, fully defensible claim:
real datasets, real models, real GPU numbers, real baselines. Mixing that
with synthetic multi-layer data in the same headline results would weaken
the credibility of both. These extensions are included to show a concrete,
buildable path toward full-stack AIOps RCA — not as validated findings.

## Path to making this real

This should be rebuilt on top of production-grade telemetry infrastructure
rather than the hand-rolled collectors currently here:

- **OpenTelemetry** for unified cross-layer telemetry collection (traces,
  metrics, logs) instead of ad hoc `subprocess` calls to `ipmitool`
- **Redfish API / Ansible's hardware management modules** for real BMC
  interaction instead of raw `ipmitool` wrapping
- A real bare-metal test lab with actual boot-trust and provisioning
  failure injection (e.g., an intentionally unsigned bootloader, an
  intentionally misconfigured DHCP scope) rather than narrative text
  generation

If pursued further, this belongs in **its own follow-up paper** once
validated on real infrastructure — not bundled into the core benchmark's
results section.
