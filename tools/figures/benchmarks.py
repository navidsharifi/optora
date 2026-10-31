"""The benchmarks page: latency, scaling, batching, host synchronizations, references.

`benchmarks-data` measures everything once and writes `benchmarks-results.json`
plus a Markdown fragment the page includes; `benchmarks-charts` draws that
data once per colour scheme. The timings are those of the machine building
the documentation (the CPU of the GitHub-hosted runner in CI); the page
says so and shows how to reproduce them on any device.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

import optora
from optora.core import AmbiguitySet
from optora.divergences import PhiDivergence
from optora.dro import PhiAmbiguitySet
from optora.solvers import GradientDescent
from tools.figures._base import SET_LABELS, Context, FigureSpec, set_color
from tools.figures.problems import Instance, cast, random_instance
from tools.figures.sets import BANDWIDTH, FAMILIES, FAMILY_BY_KEY

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.dro import oracles  # noqa: E402

RESULTS_FILE = "benchmarks-results.json"
SUMMARY_FILE = "benchmarks-summary.md"
BATCH = 16
SUPPORT = 256
BUDGET_FRACTION = 0.3
PHI_MAX_ITER = 200
PHI_CHECK_INTERVAL = 50
BAR = 0.36
# Sets whose worst case is a fixed, dtype-determined sequence of kernels:
# zero host synchronizations, forward and backward, enforced by the test suite.
ZERO_SYNC_SETS = (
    "kl",
    "chi_square",
    "total_variation",
    "cvar",
    "wasserstein",
    "sinkhorn",
)

REFERENCES: dict[str, tuple[str, Callable[..., float]]] = {
    "kl": ("SciPy `brentq` on the closed-form tilt", oracles.kl_worst_case),
    "chi_square": ("closed form (active set)", oracles.chi_square_worst_case),
    "total_variation": ("SciPy `linprog` (HiGHS)", oracles.total_variation_worst_case),
    "cvar": ("SciPy `linprog` (HiGHS)", oracles.cvar_worst_case),
    "wasserstein": ("SciPy `linprog` (HiGHS)", oracles.wasserstein_worst_case),
    "mmd": ("Clarabel interior point", oracles.mmd_worst_case),
}


@dataclass(frozen=True, slots=True)
class Workload:
    """One ambiguity set with its benchmark label.

    Attributes:
        key: Palette key.
        build: Constructor taking an instance and a radius.
    """

    key: str
    build: Callable[[Instance, float], AmbiguitySet]


def _phi_set(instance: Instance, radius: float) -> AmbiguitySet:
    return PhiAmbiguitySet(
        instance.nominal,
        divergence=PhiDivergence(phi=lambda t: t * torch.log(t) - t + 1.0),
        radius=radius,
        phi_conjugate=lambda s: torch.exp(s) - 1.0,
        dual_solver=GradientDescent(
            step_size=0.1,
            max_iter=PHI_MAX_ITER,
            tol=1e-30,
            check_interval=PHI_CHECK_INTERVAL,
        ),
    )


WORKLOADS: tuple[Workload, ...] = tuple(
    Workload(family.key, family.build) for family in FAMILIES
) + (Workload("phi", _phi_set),)


def radius_for(key: str, instance: Instance) -> float:
    """Return the benchmark radius: a fixed fraction of the family's saturation."""
    family = FAMILY_BY_KEY.get(key, FAMILY_BY_KEY["kl"])
    return BUDGET_FRACTION * family.saturation(instance)


def support_sizes(key: str, quick: bool) -> list[int]:
    """Return the support sizes of the scaling sweep for one workload."""
    if quick:
        return [32, 64]
    if key in ("sinkhorn", "mmd", "phi"):
        return [64, 256, 1024]
    return [64, 256, 1024, 4096]


def median_seconds(function: Callable[[], Any], quick: bool) -> float:
    """Return the median wall time of one call of `function` in seconds.

    Calls are timed in blocks long enough to resolve the clock, after two
    warm-up calls, until the run has lasted `min_run_time`; the result is
    the median over blocks of the mean time per call. The process's thread
    settings are left untouched on purpose: `torch.utils.benchmark.Timer`
    switches `torch.set_num_threads` around each measurement, which on
    Windows MKL builds makes every later batched `lu_solve` fail (see
    `progress/notes.md`), and the thread count a user runs with is the one
    worth measuring.
    """
    min_run_time = 0.02 if quick else 0.4
    block_time = 0.002 if quick else 0.02
    for _ in range(2):
        function()
    calls = 1
    while True:
        started = time.perf_counter()
        for _ in range(calls):
            function()
        if time.perf_counter() - started >= block_time:
            break
        calls *= 2
    per_block: list[float] = []
    deadline = time.perf_counter() + min_run_time
    while not per_block or time.perf_counter() < deadline:
        started = time.perf_counter()
        for _ in range(calls):
            function()
        per_block.append((time.perf_counter() - started) / calls)
    return statistics.median(per_block)


@contextmanager
def host_sync_counter() -> Iterator[list[None]]:
    """Count tensor-to-host conversions, the synchronizations a solve must avoid.

    Mirrors the `host_sync_counter` fixture in `tests/conftest.py`, which pins
    the same budgets in the test suite.
    """
    syncs: list[None] = []
    conversions = {
        "__bool__": torch.Tensor.__bool__,
        "__int__": torch.Tensor.__int__,
        "__float__": torch.Tensor.__float__,
    }

    def counting(original: Callable[[torch.Tensor], Any]) -> Any:
        def wrapper(tensor: torch.Tensor) -> Any:
            syncs.append(None)
            return original(tensor)

        return wrapper

    with ExitStack() as stack:
        for name, original in conversions.items():
            stack.enter_context(patch.object(torch.Tensor, name, counting(original)))
        yield syncs


def count_host_syncs(function: Callable[[], Any]) -> int:
    """Return how many host synchronizations one call of `function` performs."""
    with host_sync_counter() as syncs:
        function()
    return len(syncs)


def environment(device: torch.device) -> dict[str, str]:
    """Describe the machine and software the numbers were measured on."""
    cpu = platform.processor() or platform.machine()
    try:
        import cpuinfo

        cpu = cpuinfo.get_cpu_info().get("brand_raw", cpu)
    except Exception:
        pass
    accelerator = "none (CPU only)"
    if device.type == "cuda":
        accelerator = torch.cuda.get_device_name(device)
    commit = "unknown"
    with suppress(OSError, subprocess.CalledProcessError):
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    return {
        "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "optora": f"{optora.__version__} ({commit})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": (
            f"{cpu} ({os.cpu_count()} logical cores, "
            f"{torch.get_num_threads()} torch threads)"
        ),
        "accelerator": accelerator,
        "runner": "GitHub Actions" if os.environ.get("GITHUB_ACTIONS") else "local",
    }


def measure(context: Context) -> dict[str, Any]:
    """Run every benchmark and return the results."""
    quick, device = context.quick, context.device
    batch, support = (2, 32) if quick else (BATCH, SUPPORT)
    generator = torch.Generator().manual_seed(0)
    results: dict[str, Any] = {
        "environment": environment(device),
        "workload": {"batch": batch, "support": support, "dtype": "float32"},
        "latency": [],
        "scaling": [],
        "throughput": [],
        "reference": [],
    }

    instance64 = random_instance(support)
    instance = cast(instance64, torch.float32)
    loss = torch.randn(batch, support, generator=generator).to(device)
    for workload in WORKLOADS:
        ambiguity_set = workload.build(instance, radius_for(workload.key, instance64))
        ambiguity_set.to(device)
        attached = loss.clone().requires_grad_()

        def forward(ambiguity_set: AmbiguitySet = ambiguity_set) -> torch.Tensor:
            return ambiguity_set.worst_case_expectation(loss)

        def backward(
            ambiguity_set: AmbiguitySet = ambiguity_set,
            attached: torch.Tensor = attached,
        ) -> torch.Tensor:
            value = ambiguity_set.worst_case_expectation(attached).sum()
            (gradient,) = torch.autograd.grad(value, attached)
            return gradient

        results["latency"].append(
            {
                "set": workload.key,
                "label": SET_LABELS[workload.key],
                "forward_ms": 1e3 * median_seconds(forward, quick),
                "backward_ms": 1e3 * median_seconds(backward, quick),
                "host_syncs_forward": count_host_syncs(forward),
                "host_syncs_backward": count_host_syncs(backward),
            }
        )
        print(f"  latency {workload.key}", flush=True)

    for workload in WORKLOADS:
        for size in support_sizes(workload.key, quick):
            sized64 = random_instance(size)
            sized = cast(sized64, torch.float32)
            ambiguity_set = workload.build(sized, radius_for(workload.key, sized64)).to(
                device
            )
            single = torch.randn(size, generator=generator).to(device)
            seconds = median_seconds(
                lambda ambiguity_set=ambiguity_set, single=single: (
                    ambiguity_set.worst_case_expectation(single)
                ),
                quick,
            )
            results["scaling"].append(
                {"set": workload.key, "n": size, "ms": 1e3 * seconds}
            )
        print(f"  scaling {workload.key}", flush=True)

    batch_sizes = [1, 4] if quick else [1, 4, 16, 64]
    for workload in WORKLOADS:
        ambiguity_set = workload.build(
            instance, radius_for(workload.key, instance64)
        ).to(device)
        for batch_size in batch_sizes:
            batched = torch.randn(batch_size, support, generator=generator).to(device)
            seconds = median_seconds(
                lambda ambiguity_set=ambiguity_set, batched=batched: (
                    ambiguity_set.worst_case_expectation(batched)
                ),
                quick,
            )
            results["throughput"].append(
                {
                    "set": workload.key,
                    "batch": batch_size,
                    "elements_per_second": batch_size / seconds,
                }
            )
        print(f"  throughput {workload.key}", flush=True)

    reference_size = 32 if quick else 128
    reference_instance = random_instance(reference_size, seed=1)
    nominal = reference_instance.nominal.numpy()
    ref_loss = reference_instance.loss.numpy()
    cost = reference_instance.cost.numpy()
    gram = reference_instance.gram(BANDWIDTH).numpy()
    spread = float(ref_loss.max() - ref_loss.min())
    for key, (label, oracle) in REFERENCES.items():
        family = FAMILY_BY_KEY[key]
        radius = radius_for(key, reference_instance)
        ambiguity_set = family.build(reference_instance, radius)
        if key == "cvar":
            reference_call = functools.partial(
                oracle, nominal, ref_loss, float(np.exp(-radius))
            )
        elif key == "wasserstein":
            reference_call = functools.partial(oracle, nominal, cost, ref_loss, radius)
        elif key == "mmd":
            reference_call = functools.partial(oracle, nominal, gram, ref_loss, radius)
        else:
            reference_call = functools.partial(oracle, nominal, ref_loss, radius)
        reference = reference_call()
        with torch.no_grad():
            value = float(ambiguity_set.worst_case_expectation(reference_instance.loss))
        results["reference"].append(
            {
                "set": key,
                "label": SET_LABELS[key],
                "reference": label,
                "n": reference_size,
                "optora_ms": 1e3
                * median_seconds(
                    lambda ambiguity_set=ambiguity_set: (
                        ambiguity_set.worst_case_expectation(reference_instance.loss)
                    ),
                    quick,
                ),
                "reference_ms": 1e3 * median_seconds(reference_call, quick),
                "agreement": abs(value - reference) / spread,
            }
        )
        print(f"  reference {key}", flush=True)
    return results


def _stat(label: str, value: str, note: str) -> str:
    return (
        '<div class="stat">'
        f'<span class="stat__label">{label}</span>'
        f'<span class="stat__value">{value}</span>'
        f'<span class="stat__note">{note}</span>'
        "</div>"
    )


def write_summary(results: dict[str, Any], path: Path) -> None:
    """Write the Markdown fragment the benchmarks page includes."""
    latency = results["latency"]
    exact = [row for row in latency if row["set"] != "phi"]
    fastest = min(exact, key=lambda row: row["forward_ms"])
    throughput = max(results["throughput"], key=lambda row: row["elements_per_second"])
    agreement = max(row["agreement"] for row in results["reference"])
    largest = max(row["n"] for row in results["scaling"])
    syncs = max(
        row["host_syncs_backward"] for row in exact if row["set"] in ZERO_SYNC_SETS
    )
    workload = results["workload"]
    lines = [
        '<div class="stat-grid">',
        _stat(
            "Host synchronizations",
            str(syncs),
            "per tilt- or closed-form-solved worst case, forward and backward",
        ),
        _stat(
            "Fastest worst case",
            f"{fastest['forward_ms']:.2f} ms",
            f"{fastest['label']}, batch {workload['batch']} x "
            f"{workload['support']} float32",
        ),
        _stat(
            "Batched throughput",
            f"{throughput['elements_per_second']:,.0f} /s",
            f"{SET_LABELS[throughput['set']]} loss vectors of "
            f"n = {workload['support']} per second at batch {throughput['batch']}",
        ),
        _stat(
            "Agreement with references",
            f"{agreement:.1e}",
            "largest |difference| / loss spread, float64",
        ),
        "</div>",
        "",
        "| Environment | |",
        "| --- | --- |",
    ]
    for key, value in results["environment"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "| Ambiguity set | Forward (ms) | Forward + backward (ms) | Host syncs "
        "| Reference solver | Agreement | Speedup |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    reference_by_set = {row["set"]: row for row in results["reference"]}
    for row in latency:
        reference = reference_by_set.get(row["set"])
        if reference is None:
            ref_columns = "none in the test suite | | "
        else:
            ref_columns = (
                f"{reference['reference']} | {reference['agreement']:.1e} | "
                f"{reference['reference_ms'] / reference['optora_ms']:.1f}x"
            )
        lines.append(
            f"| {row['label']} | {row['forward_ms']:.2f} | {row['backward_ms']:.2f} | "
            f"{row['host_syncs_backward']} | {ref_columns} |"
        )
    lines.append("")
    lines.append(
        f"The largest support in the scaling sweep is n = {largest}. "
        "The general phi set runs a fixed-step dual solve of "
        f"{PHI_MAX_ITER} iterations with a convergence check every "
        f"{PHI_CHECK_INTERVAL}, hence its host synchronizations."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_data(context: Context) -> None:
    """Measure the benchmarks and write the results and the summary fragment."""
    started = time.perf_counter()
    results = measure(context)
    results["elapsed_seconds"] = time.perf_counter() - started
    context.path(RESULTS_FILE).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    write_summary(results, context.path(SUMMARY_FILE))


def load_results(context: Context) -> dict[str, Any]:
    """Read the results `benchmarks-data` wrote."""
    path = context.path(RESULTS_FILE)
    if not path.exists():
        msg = f"{path} is missing; render benchmarks-data first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _plain_ticks(axes: Any, values: list[int]) -> None:
    """Label a log-scaled x axis with the plain measured values."""
    axes.set_xticks(values)
    axes.xaxis.set_major_formatter(ScalarFormatter())
    axes.xaxis.set_minor_formatter(plt.NullFormatter())


def _legend_below(axes: Any) -> None:
    """Place a legend in one row under the plot, clear of every line."""
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)


def render_charts(context: Context) -> None:
    """Draw the latency, scaling, throughput and reference charts."""
    theme = context.theme
    results = load_results(context)
    workload = results["workload"]

    latency = sorted(results["latency"], key=lambda row: row["forward_ms"])
    fig, axes = plt.subplots(figsize=(8.6, 4.2))
    positions = np.arange(len(latency))
    height = BAR
    forward = [row["forward_ms"] for row in latency]
    backward = [row["backward_ms"] for row in latency]
    axes.barh(
        positions + height / 2,
        forward,
        height=height,
        color=theme.series[0],
        label="forward",
    )
    axes.barh(
        positions - height / 2,
        backward,
        height=height,
        color=theme.series[1],
        label="forward + backward",
    )
    for position, (fwd, bwd) in enumerate(zip(forward, backward, strict=True)):
        axes.text(
            fwd * 1.08,
            position + height / 2,
            f"{fwd:.2f}",
            va="center",
            fontsize=8,
            color=theme.secondary,
        )
        axes.text(
            bwd * 1.08,
            position - height / 2,
            f"{bwd:.2f}",
            va="center",
            fontsize=8,
            color=theme.secondary,
        )
    axes.set_yticks(positions)
    axes.set_yticklabels([row["label"] for row in latency])
    axes.set_xscale("log")
    axes.set_xlim(right=max(backward) * 3)
    axes.set_xlabel("median wall time per call (ms, log)")
    axes.set_title(
        f"worst-case expectation, batch {workload['batch']} x "
        f"n = {workload['support']}, float32"
    )
    axes.legend(loc="lower right")
    axes.grid(axis="y", visible=False)
    fig.tight_layout()
    context.save_static(fig, "benchmarks-latency")

    fig, axes = plt.subplots(figsize=(8.6, 4.4))
    for key in [row.key for row in WORKLOADS]:
        rows = [row for row in results["scaling"] if row["set"] == key]
        axes.plot(
            [row["n"] for row in rows],
            [row["ms"] for row in rows],
            marker="o",
            color=set_color(theme, key),
            label=SET_LABELS[key],
        )
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    _plain_ticks(axes, sorted({row["n"] for row in results["scaling"]}))
    axes.set_xlabel("support size n (one loss vector, float32)")
    axes.set_ylabel("median wall time (ms, log)")
    axes.set_title("scaling with the support size")
    _legend_below(axes)
    fig.tight_layout()
    context.save_static(fig, "benchmarks-scaling")

    fig, axes = plt.subplots(figsize=(8.6, 4.4))
    for key in [row.key for row in WORKLOADS]:
        rows = [row for row in results["throughput"] if row["set"] == key]
        axes.plot(
            [row["batch"] for row in rows],
            [row["elements_per_second"] for row in rows],
            marker="o",
            color=set_color(theme, key),
            label=SET_LABELS[key],
        )
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    _plain_ticks(axes, sorted({row["batch"] for row in results["throughput"]}))
    axes.set_xlabel(
        f"batch size (loss vectors of n = {workload['support']} solved in one call)"
    )
    axes.set_ylabel("loss vectors per second (log)")
    axes.set_title("throughput grows with the batch: one joint solve, no Python loop")
    _legend_below(axes)
    fig.tight_layout()
    context.save_static(fig, "benchmarks-throughput")

    reference = results["reference"]
    fig, axes = plt.subplots(figsize=(8.6, 4.2))
    positions = np.arange(len(reference))
    optora_ms = [row["optora_ms"] for row in reference]
    reference_ms = [row["reference_ms"] for row in reference]
    axes.bar(
        positions - height / 2,
        optora_ms,
        width=height,
        color=theme.series[0],
        label="Optora (float64)",
    )
    axes.bar(
        positions + height / 2,
        reference_ms,
        width=height,
        color=theme.series[1],
        label="reference solver",
    )
    for position, row in enumerate(reference):
        axes.text(
            position,
            max(row["optora_ms"], row["reference_ms"]) * 1.25,
            f"{row['reference_ms'] / row['optora_ms']:.1f}x\n"
            f"|diff| {row['agreement']:.0e}",
            ha="center",
            fontsize=7.5,
            color=theme.secondary,
        )
    axes.set_xticks(positions)
    axes.set_xticklabels([row["label"] for row in reference])
    axes.set_yscale("log")
    axes.set_ylim(top=max(reference_ms) * 12)
    axes.set_ylabel("median wall time (ms, log)")
    axes.set_title(
        f"one worst case at n = {reference[0]['n']}: "
        "Optora against an independent reference solver"
    )
    axes.legend(loc="upper left")
    axes.grid(axis="x", visible=False)
    fig.tight_layout()
    context.save_static(fig, "benchmarks-reference")


FIGURES = (
    FigureSpec(
        name="benchmarks-data",
        page="benchmarks.md",
        outputs=(RESULTS_FILE, SUMMARY_FILE),
        render=render_data,
    ),
    FigureSpec(
        name="benchmarks-charts",
        page="benchmarks.md",
        outputs=(
            "benchmarks-latency-{theme}.svg",
            "benchmarks-scaling-{theme}.svg",
            "benchmarks-throughput-{theme}.svg",
            "benchmarks-reference-{theme}.svg",
        ),
        render=render_charts,
    ),
)
