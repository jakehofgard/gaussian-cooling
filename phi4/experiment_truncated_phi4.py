"""Convenience launcher for the standalone lattice phi4 experiments.

Researchers should normally invoke the focused experiment modules directly;
each has its own documented defaults and CLI.  This small launcher retains a
one-command ordinary suite, a reduced smoke suite, and the historical bundled
GPU scaling run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence


ExperimentMain = Callable[[Sequence[str] | None], None]

ORDINARY_EXPERIMENTS = (
    "scaling",
    "stages",
    "diagnostics",
    "correlator",
    "sweeps",
)
GPU_EXPERIMENTS = ("scaling", "stages")


def _experiment_mains() -> dict[str, ExperimentMain]:
    """Import experiment entry points lazily after parsing the suite CLI."""

    from .experiment_phi4_budget_scaling import main as budget_main
    from .experiment_phi4_correlator import main as correlator_main
    from .experiment_phi4_diagnostics import main as diagnostics_main
    from .experiment_phi4_hardness_map import main as hardness_main
    from .experiment_phi4_parameter_sweeps import main as sweeps_main
    from .experiment_phi4_scaling import main as scaling_main
    from .experiment_phi4_stage_convergence import main as stages_main

    return {
        "scaling": scaling_main,
        "stages": stages_main,
        "budget": budget_main,
        "diagnostics": diagnostics_main,
        "correlator": correlator_main,
        "sweeps": sweeps_main,
        "hardness": hardness_main,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the suite selector; scientific options belong to focused modules."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the lattice phi4 experiment suite. Use the focused "
            "phi4.experiment_phi4_* modules for scientific parameter changes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=(*ORDINARY_EXPERIMENTS, "budget", "hardness"),
        action="append",
        help=(
            "Run only the selected workflow; repeat for several. By default "
            "run the five ordinary workflows."
        ),
    )
    parser.add_argument(
        "--hardness-map",
        "--hardness-map-only",
        action="store_true",
        help="Run only the fixed-R, fixed-beta hardness map.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Apply each selected experiment's CPU smoke-test preset.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Require a GPU. Without --only, run lattice scaling plus stage "
            "convergence, matching the historical H100 suite; budget "
            "scaling is also available through --only budget."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        "--output",
        dest="output",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "figures" / "truncated_phi4"),
        help="Shared base prefix; every experiment appends its PDF suffix.",
    )
    return parser


def _selected_experiments(args: argparse.Namespace) -> tuple[str, ...]:
    """Resolve suite presets and selections, preserving requested run order."""

    if args.quick and args.gpu:
        raise ValueError("--quick and --gpu are mutually exclusive presets.")
    if args.hardness_map:
        if args.only:
            raise ValueError("--hardness-map cannot be combined with --only.")
        return ("hardness",)
    if args.only:
        selected = tuple(dict.fromkeys(args.only))
        gpu_compatible = (*GPU_EXPERIMENTS, "budget", "hardness")
        if args.gpu and any(name not in gpu_compatible for name in selected):
            raise ValueError(
                "The suite --gpu preset supports scaling, stages, budget "
                "scaling, and the hardness-map GPU check; invoke other "
                "focused modules directly."
            )
        return selected
    return GPU_EXPERIMENTS if args.gpu else ORDINARY_EXPERIMENTS


def main(argv: Sequence[str] | None = None) -> None:
    """Run selected workflows with the shared preset and output prefix."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        selected = _selected_experiments(args)
    except ValueError as exc:
        parser.error(str(exc))

    experiment_mains = _experiment_mains()
    forwarded = ["--output-prefix", str(args.output)]
    if args.quick:
        forwarded.append("--quick")
    if args.gpu:
        forwarded.append("--gpu")

    print(
        "Lattice phi4 suite: " + ", ".join(selected),
        flush=True,
    )
    for name in selected:
        print(f"\n=== {name} experiment ===", flush=True)
        experiment_arguments = list(forwarded)
        if name == "stages" and not args.quick and not args.gpu:
            experiment_arguments.extend(["--key-side-order", "5,10,20,50,100"])
        experiment_mains[name](experiment_arguments)


if __name__ == "__main__":
    main()
