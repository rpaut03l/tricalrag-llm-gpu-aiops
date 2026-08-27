"""
WHY multiple prompt variants: one of our ablations tests whether few-shot
examples improve RCA accuracy enough to justify the extra tokens/latency —
a genuinely useful finding for practitioners deciding how to deploy this.
"""

ZERO_SHOT_TEMPLATE = """You are an SRE assistant analyzing system logs to detect and explain incidents.

Below is a window of raw log lines from a production system:

---
{log_window}
---

Analyze this log window and respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "is_anomaly": true or false,
  "severity": "low" | "medium" | "high" | "critical",
  "root_cause": "one sentence explanation of what went wrong, or 'none' if normal",
  "remediation": "one sentence suggested fix, or 'none' if normal"
}}
"""

FEW_SHOT_TEMPLATE = """You are an SRE assistant analyzing system logs to detect and explain incidents.

Example 1:
Log window:
INFO dfs.DataNode: Received block blk_123 of size 512
INFO dfs.DataNode: Served block blk_123 to /10.0.0.5
Answer: {{"is_anomaly": false, "severity": "low", "root_cause": "none", "remediation": "none"}}

Example 2:
Log window:
ERROR dfs.DataNode: Exception in receiveBlock for block blk_456
ERROR dfs.DataNode: java.io.IOException: Connection reset by peer
WARN dfs.DataNode: Retrying block transfer blk_456
Answer: {{"is_anomaly": true, "severity": "high", "root_cause": "Network connection was reset during block transfer, likely due to a transient network fault or overloaded DataNode.", "remediation": "Increase retry timeout and monitor network stability between DataNodes."}}

Now analyze this log window:
---
{log_window}
---

Respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "is_anomaly": true or false,
  "severity": "low" | "medium" | "high" | "critical",
  "root_cause": "one sentence explanation of what went wrong, or 'none' if normal",
  "remediation": "one sentence suggested fix, or 'none' if normal"
}}
"""

FEW_SHOT_TEMPLATE = """You are an SRE assistant analyzing system logs to detect and explain incidents.

Example 1:
Log window:
INFO dfs.DataNode: Received block blk_123 of size 512
INFO dfs.DataNode: Served block blk_123 to /10.0.0.5
Answer: {{"is_anomaly": false, "severity": "low", "root_cause": "none", "remediation": "none"}}

Example 2:
Log window:
ERROR dfs.DataNode: Exception in receiveBlock for block blk_456
ERROR dfs.DataNode: java.io.IOException: Connection reset by peer
WARN dfs.DataNode: Retrying block transfer blk_456
Answer: {{"is_anomaly": true, "severity": "high", "root_cause": "Network connection was reset during block transfer, likely due to a transient network fault or overloaded DataNode.", "remediation": "Increase retry timeout and monitor network stability between DataNodes."}}

Now analyze this log window:
---
{log_window}
---

Respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "is_anomaly": true or false,
  "severity": "low" | "medium" | "high" | "critical",
  "root_cause": "one sentence explanation of what went wrong, or 'none' if normal",
  "remediation": "one sentence suggested fix, or 'none' if normal"
}}
"""

RAG_TEMPLATE = """You are an SRE assistant analyzing system logs to detect and explain incidents.
You have access to similar past incidents retrieved from an incident history database.
Use them as precedent, but base your final judgment on the CURRENT log window.

--- RETRIEVED PAST INCIDENTS ---
{retrieved_context}
--- END RETRIEVED CONTEXT ---

Now analyze this NEW log window:
---
{log_window}
---

Respond with ONLY a JSON object (no extra text) in this exact format:
{{
  "is_anomaly": true or false,
  "severity": "low" | "medium" | "high" | "critical",
  "root_cause": "one sentence explanation of what went wrong, or 'none' if normal",
  "remediation": "one sentence suggested fix, or 'none' if normal"
}}
"""

def build_prompt(log_window: str, style: str = "zero_shot", retrieved_context: str = None) -> str:
    if style == "rag":
        return RAG_TEMPLATE.format(log_window=log_window, retrieved_context=retrieved_context or "None found.")
    if style == "few_shot":
        return FEW_SHOT_TEMPLATE.format(log_window=log_window)
    return ZERO_SHOT_TEMPLATE.format(log_window=log_window)
