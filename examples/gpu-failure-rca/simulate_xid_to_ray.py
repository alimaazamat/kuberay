#!/usr/bin/env python3
"""
Simulate how a GPU-level (Xid) fault propagates up the stack and surfaces
as an "application-level" failure in Ray, and how NPD + an Xid-aware
correlator turns the same event into a clear root-cause attribution.

No real GPU is needed. Every layer is a small in-memory simulator:

    GPU hardware fault
        -> NVIDIA driver writes Xid to kernel log (/dev/kmsg)
        -> Node Problem Detector (NPD) tails kernel log,
           creates NodeCondition + Event
        -> Kubelet / control plane mark node unhealthy
        -> Ray worker process (running on that node) hits a CUDA error
        -> Ray reports task / actor / job failure
        -> User sees "application-level" failure

Run two scenarios:

    Scenario A (no NPD, no Xid awareness):
        Ray surfaces RayTaskError("CUDA error: unspecified launch failure")
        and the user concludes "app bug".

    Scenario B (NPD + Xid correlator enabled):
        Same underlying fault, but the correlator joins the Ray failure
        timestamp + node with the NPD/Xid event and reports the true
        root cause: GPU Xid 79 (GPU has fallen off the bus).

Usage:
    python3 simulate_xid_to_ray.py
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import random
import re
import textwrap
import time
from typing import Callable


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


@dataclasses.dataclass
class KernelLogLine:
    ts: dt.datetime
    node: str
    facility: str  # e.g. "kern", "NVRM"
    message: str

    def __str__(self) -> str:
        return f"[{self.ts.isoformat(timespec='seconds')}] {self.node} kernel: {self.message}"


@dataclasses.dataclass
class NodeCondition:
    type: str
    status: str  # "True" / "False" / "Unknown"
    reason: str
    message: str
    last_transition: dt.datetime


@dataclasses.dataclass
class NodeEvent:
    ts: dt.datetime
    node: str
    type: str  # "Warning" / "Normal"
    reason: str
    message: str
    source: str  # "npd" / "kubelet"


@dataclasses.dataclass
class RayFailure:
    ts: dt.datetime
    job_id: str
    task_id: str
    node: str
    error_type: str  # "RayTaskError", "WorkerCrashedError", ...
    message: str


# ---------------------------------------------------------------------------
# Layer 1: GPU + NVIDIA driver
# ---------------------------------------------------------------------------


# A small subset of real Xid codes. See:
#   https://docs.nvidia.com/deploy/xid-errors/index.html
XID_CATALOG = {
    13: ("Graphics Engine Exception", "app"),
    31: ("GPU memory page fault", "app_or_driver"),
    43: ("GPU stopped processing", "app"),
    63: ("ECC page retirement", "hardware"),
    64: ("ECC double-bit error", "hardware"),
    74: ("NVLink error", "hardware"),
    79: ("GPU has fallen off the bus", "hardware"),
}


class GPU:
    """Simulated GPU. Can be told to fault at the next CUDA call."""

    def __init__(self, node: str, index: int = 0):
        self.node = node
        self.index = index
        self._pending_xid: int | None = None

    def inject_fault(self, xid: int) -> None:
        self._pending_xid = xid

    def cuda_launch(self) -> None:
        """Pretend to run a CUDA kernel. Raises if a fault is pending."""
        if self._pending_xid is None:
            return
        xid = self._pending_xid
        # Driver emits Xid to kernel log *before* the user-space CUDA
        # call returns an error (this is how it works on real systems).
        KernelLog.instance().emit(
            KernelLogLine(
                ts=now(),
                node=self.node,
                facility="NVRM",
                message=(
                    f"NVRM: Xid (PCI:0000:00:{self.index:02d}): {xid}, "
                    f"pid=12345, {XID_CATALOG[xid][0]}"
                ),
            )
        )
        # Map Xid -> CUDA-level exception the user code actually sees.
        # On a real system the driver returns a generic CUDA error code;
        # the Xid in dmesg is the only place the *real* cause is named.
        if xid == 79:
            # GPU fell off the bus -> next call returns "unspecified launch failure"
            raise RuntimeError("CUDA error: unspecified launch failure")
        if xid in (63, 64):
            raise RuntimeError("CUDA error: uncorrectable ECC error encountered")
        if xid == 74:
            raise RuntimeError("CUDA error: NCCL communication error")
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


# ---------------------------------------------------------------------------
# Layer 2: Kernel log (/dev/kmsg, journalctl -k)
# ---------------------------------------------------------------------------


class KernelLog:
    _instance: "KernelLog | None" = None

    def __init__(self) -> None:
        self.lines: list[KernelLogLine] = []
        self._subscribers: list[Callable[[KernelLogLine], None]] = []

    @classmethod
    def instance(cls) -> "KernelLog":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def emit(self, line: KernelLogLine) -> None:
        self.lines.append(line)
        for cb in self._subscribers:
            cb(line)

    def subscribe(self, cb: Callable[[KernelLogLine], None]) -> None:
        self._subscribers.append(cb)


# ---------------------------------------------------------------------------
# Layer 3: Node Problem Detector
# ---------------------------------------------------------------------------


class NodeProblemDetector:
    """Tails the kernel log, matches Xid patterns, sets NodeConditions
    and emits Kubernetes Events. Mirrors the kernel-monitor plugin's
    behavior with a custom Xid rule set.
    """

    XID_REGEX = re.compile(r"NVRM:\s*Xid\s*\(PCI:[^)]+\):\s*(\d+),")

    def __init__(self, kapi: "KubeAPI", enabled: bool = True):
        self.kapi = kapi
        self.enabled = enabled
        if enabled:
            KernelLog.instance().subscribe(self._on_line)

    def _on_line(self, line: KernelLogLine) -> None:
        m = self.XID_REGEX.search(line.message)
        if not m:
            return
        xid = int(m.group(1))
        desc, severity = XID_CATALOG.get(xid, ("Unknown Xid", "unknown"))

        # Permanent condition (e.g. Xid 79, 63, 64, 74) -> NodeCondition.
        permanent = severity == "hardware"
        if permanent:
            self.kapi.set_condition(
                line.node,
                NodeCondition(
                    type="GPUHealthy",
                    status="False",
                    reason=f"Xid{xid}",
                    message=f"NVIDIA Xid {xid}: {desc}",
                    last_transition=line.ts,
                ),
            )
        # Always emit a Warning event so the correlator can find it.
        self.kapi.emit_event(
            NodeEvent(
                ts=line.ts,
                node=line.node,
                type="Warning",
                reason=f"NvidiaXid{xid}",
                message=f"Xid {xid} ({desc}) detected on {line.node}",
                source="npd",
            )
        )


# ---------------------------------------------------------------------------
# Layer 4: Kubernetes API (very small simulator)
# ---------------------------------------------------------------------------


class KubeAPI:
    def __init__(self) -> None:
        self.node_conditions: dict[str, list[NodeCondition]] = {}
        self.events: list[NodeEvent] = []

    def set_condition(self, node: str, cond: NodeCondition) -> None:
        conds = self.node_conditions.setdefault(node, [])
        conds[:] = [c for c in conds if c.type != cond.type]
        conds.append(cond)

    def emit_event(self, ev: NodeEvent) -> None:
        self.events.append(ev)

    def get_events_for_node(
        self, node: str, since: dt.datetime, until: dt.datetime
    ) -> list[NodeEvent]:
        return [e for e in self.events if e.node == node and since <= e.ts <= until]


# ---------------------------------------------------------------------------
# Layer 5: Ray worker / job
# ---------------------------------------------------------------------------


class RayWorker:
    def __init__(self, node: str, gpu: GPU):
        self.node = node
        self.gpu = gpu

    def run_task(self, job_id: str, task_id: str) -> RayFailure | None:
        """Run a CUDA task. Return None on success, RayFailure on error."""
        try:
            self.gpu.cuda_launch()
            return None
        except RuntimeError as e:
            # This is what Ray actually surfaces to the user. The driver-level
            # Xid is *not* in this message -- it only lives in dmesg.
            return RayFailure(
                ts=now(),
                job_id=job_id,
                task_id=task_id,
                node=self.node,
                error_type="RayTaskError",
                message=(
                    f"ray.exceptions.RayTaskError(RuntimeError): "
                    f"\u001b[36mray::train_step()\u001b[39m (pid=12345, "
                    f"ip=10.0.0.4) {e}"
                ),
            )


# ---------------------------------------------------------------------------
# Layer 6: Correlator (the new piece)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RCAReport:
    failure: RayFailure
    likely_app_failure: bool
    matched_events: list[NodeEvent]
    node_conditions: list[NodeCondition]
    verdict: str
    explanation: str


class XidCorrelator:
    """Given a Ray failure, look at the node it ran on within +/- window
    and decide whether the "application failure" was actually caused by a
    GPU-level Xid event surfaced by NPD.
    """

    WINDOW = dt.timedelta(seconds=30)

    # CUDA-level error substrings that *could* be GPU-caused. Without NPD
    # these are ambiguous; with NPD we can disambiguate.
    AMBIGUOUS_CUDA_PATTERNS = (
        "CUDA error: unspecified launch failure",
        "CUDA error: an illegal memory access",
        "CUDA error: uncorrectable ECC",
        "NCCL communication error",
        "CUDA driver",
    )

    def __init__(self, kapi: KubeAPI):
        self.kapi = kapi

    def explain(self, failure: RayFailure) -> RCAReport:
        ambiguous = any(p in failure.message for p in self.AMBIGUOUS_CUDA_PATTERNS)
        events = self.kapi.get_events_for_node(
            failure.node,
            since=failure.ts - self.WINDOW,
            until=failure.ts + self.WINDOW,
        )
        xid_events = [e for e in events if e.reason.startswith("NvidiaXid")]
        conds = [
            c
            for c in self.kapi.node_conditions.get(failure.node, [])
            if c.type == "GPUHealthy" and c.status == "False"
        ]

        if xid_events:
            xid_event = xid_events[0]
            verdict = "GPU/HARDWARE ROOT CAUSE"
            explanation = (
                f"Ray reported {failure.error_type} on node {failure.node}, "
                f"but NPD recorded {xid_event.reason} "
                f"({xid_event.message}) {(failure.ts - xid_event.ts).total_seconds():.1f}s "
                f"before the Ray failure. The CUDA error surfaced to the user is "
                f"a downstream symptom of the Xid event, not an application bug."
            )
            return RCAReport(
                failure=failure,
                likely_app_failure=False,
                matched_events=xid_events,
                node_conditions=conds,
                verdict=verdict,
                explanation=explanation,
            )

        if ambiguous:
            verdict = "AMBIGUOUS - LIKELY APP, COULD BE GPU"
            explanation = (
                "Ray surfaced a CUDA-level error, but no NPD/Xid signal was "
                "found near the failure. Either NPD is not running with Xid "
                "rules, or this really is an application bug."
            )
        else:
            verdict = "APPLICATION FAILURE"
            explanation = "No GPU signals near this failure; treat as app-level."

        return RCAReport(
            failure=failure,
            likely_app_failure=True,
            matched_events=[],
            node_conditions=conds,
            verdict=verdict,
            explanation=explanation,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n  {title}\n{bar}")
    else:
        print(bar)


def show_dmesg() -> None:
    print("\n# journalctl -k | grep -i nvrm")
    for line in KernelLog.instance().lines:
        if line.facility == "NVRM":
            print(f"  {line}")
    if not any(l.facility == "NVRM" for l in KernelLog.instance().lines):
        print("  (no NVRM entries)")


def show_node(kapi: KubeAPI, node: str) -> None:
    print(f"\n# kubectl describe node {node}")
    conds = kapi.node_conditions.get(node, [])
    if not conds:
        print("  Conditions: (none related to GPU)")
    else:
        print("  Conditions:")
        for c in conds:
            print(
                f"    - type={c.type} status={c.status} reason={c.reason} "
                f"message={c.message!r}"
            )
    evs = [e for e in kapi.events if e.node == node]
    print("  Events:")
    if not evs:
        print("    (none)")
    for e in evs:
        print(
            f"    - {e.ts.isoformat(timespec='seconds')} {e.type} "
            f"{e.reason} ({e.source}): {e.message}"
        )


def show_ray_failure(f: RayFailure) -> None:
    print("\n# ray job logs raysubmit_xxxx   (what the user sees)")
    print(f"  Job:  {f.job_id}")
    print(f"  Task: {f.task_id}  on node {f.node}")
    print(f"  Time: {f.ts.isoformat(timespec='seconds')}")
    print(f"  Type: {f.error_type}")
    print(textwrap.indent(f.message, "    "))


def show_rca(report: RCAReport) -> None:
    print("\n# kuberay-rca explain --job ...")
    print(f"  Verdict     : {report.verdict}")
    print(f"  Likely app? : {report.likely_app_failure}")
    print(f"  Explanation : {report.explanation}")
    if report.matched_events:
        print("  Matched NPD events:")
        for e in report.matched_events:
            print(f"    - {e.reason}: {e.message}")
    if report.node_conditions:
        print("  Node conditions:")
        for c in report.node_conditions:
            print(f"    - {c.type}={c.status} reason={c.reason}")


def run_scenario(label: str, npd_enabled: bool, xid: int) -> None:
    hr(f"SCENARIO {label}: npd_enabled={npd_enabled}, injected Xid={xid}")

    # Reset global state between scenarios.
    KernelLog._instance = None

    kapi = KubeAPI()
    NodeProblemDetector(kapi, enabled=npd_enabled)

    node = "aks-gpunp-12345-vmss000000"
    gpu = GPU(node=node, index=0)
    worker = RayWorker(node=node, gpu=gpu)

    # 1. Run a healthy task first to show baseline.
    print("\n[step 1] Ray runs train_step() -- healthy")
    failure = worker.run_task(job_id="raysubmit_001", task_id="train_step.0")
    print("  result: OK" if failure is None else f"  result: {failure.error_type}")

    # 2. Hardware fault occurs (in real life: power glitch, thermal,
    #    PCIe link drop, ECC, NVLink, etc.)
    print(f"\n[step 2] GPU on {node} faults -- driver will emit Xid {xid}")
    gpu.inject_fault(xid)
    time.sleep(0.01)

    # 3. Ray runs the next task on the bad GPU.
    print("\n[step 3] Ray runs train_step() again on the same GPU")
    failure = worker.run_task(job_id="raysubmit_001", task_id="train_step.1")
    assert failure is not None

    # 4. What the user / dashboard sees.
    show_dmesg()
    show_node(kapi, node)
    show_ray_failure(failure)

    # 5. Run the correlator.
    correlator = XidCorrelator(kapi)
    report = correlator.explain(failure)
    show_rca(report)


def main() -> None:
    random.seed(0)

    hr("KubeRay + NPD + Xid: how a GPU fault becomes an 'app failure'")
    print(
        textwrap.dedent(
            """
            We simulate the full chain:

              GPU hardware fault
                -> NVIDIA driver writes Xid to kernel log
                -> NPD (with Xid rules) sets NodeCondition + Event
                -> Ray worker's CUDA call raises RuntimeError
                -> Ray surfaces RayTaskError to the user

            Scenario A turns NPD off, so only the CUDA-level error is visible
            and the failure looks application-level.

            Scenario B turns NPD on with Xid rules, so a correlator can
            attribute the same Ray failure to the underlying GPU Xid.
            """
        ).strip()
    )

    run_scenario("A (no NPD / no Xid awareness)", npd_enabled=False, xid=79)
    run_scenario("B (NPD + Xid rules + correlator)", npd_enabled=True, xid=79)

    hr("Takeaway")
    print(
        textwrap.dedent(
            """
            Same underlying fault. Same Ray error message. Different RCA:

              A) Without NPD/Xid: user sees 'RayTaskError: CUDA error:
                 unspecified launch failure' and concludes the training
                 script has a bug. Real cause (Xid 79 - GPU fell off the
                 bus) is invisible outside the kernel log.

              B) With NPD/Xid: the same Ray failure is automatically tied
                 to the NPD Xid event on the node it ran on, and the
                 correlator reports 'GPU/HARDWARE ROOT CAUSE' with the
                 specific Xid code -- no human triage needed.

            For a real KubeRay deployment on AKS today, the equivalent
            pieces are:

              * AKS NPD DaemonSet with a kernel-monitor rule for
                'NVRM: Xid \\\\(PCI:[^)]+\\\\): (\\\\d+),' setting
                NodeCondition GPUHealthy=False and a Warning event.
              * A stand-alone correlator script (see correlate.py in the
                README) that, given a failed RayJob, lists Events on the
                pod's node within +/- 30s and prints the matched Xid.

            See FUTURE_WORK.md for a sketch of how this could be folded
            into the KubeRay operator itself.
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
