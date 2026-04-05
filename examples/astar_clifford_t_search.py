r"""Best-first search for Clifford+T approximations of dyadic Z rotations.

This script searches over one-qubit gate sequences built from H, S, and T
to approximate the family

    Rz(pi / 2**n),  n in {0, 1, 2, 3, 4, 5}

using the global-phase-invariant distance metric from challenge.md:

    d(U, V) = sqrt(1 - |Tr(U^\dagger V)| / 2)

The search is intentionally simple:

- every gate counts the same
- T is not given any special weight
- the frontier is ordered by current distance to the target
- shorter sequences break ties

The script also supports a threshold sweep mode so you can tighten the target
distance by orders of magnitude and inspect the best final distance reached.
"""

from __future__ import annotations

import argparse
import cmath
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SQRT2 = math.sqrt(2.0)
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]
IDENTITY: Matrix2 = ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))


def rz(theta: float) -> Matrix2:
    """Return the single-qubit Z rotation Rz(theta)."""
    return (
        (cmath.exp(-1j * theta / 2.0), 0.0j),
        (0.0j, cmath.exp(1j * theta / 2.0)),
    )


GATES: Dict[str, Matrix2] = {
    "H": ((1.0 / SQRT2, 1.0 / SQRT2), (1.0 / SQRT2, -1.0 / SQRT2)),
    "S": ((1.0 + 0.0j, 0.0j), (0.0j, 1.0j)),
    "T": ((1.0 + 0.0j, 0.0j), (0.0j, cmath.exp(1j * math.pi / 4.0))),
}


def matmul(a: Matrix2, b: Matrix2) -> Matrix2:
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def dagger(a: Matrix2) -> Matrix2:
    return (
        (a[0][0].conjugate(), a[1][0].conjugate()),
        (a[0][1].conjugate(), a[1][1].conjugate()),
    )


def trace(a: Matrix2) -> complex:
    return a[0][0] + a[1][1]


def challenge_distance(target: Matrix2, candidate: Matrix2) -> float:
    """Global-phase-invariant distance from the challenge statement."""
    overlap = abs(trace(matmul(dagger(target), candidate))) / 2.0
    overlap = min(1.0, max(0.0, float(overlap)))
    return math.sqrt(1.0 - overlap)


def remove_global_phase(unitary: Matrix2, eps: float = 1e-12) -> Matrix2:
    flat = (unitary[0][0], unitary[0][1], unitary[1][0], unitary[1][1])
    for value in flat:
        if abs(value) > eps:
            phase = value / abs(value)
            return tuple(
                tuple(entry / phase for entry in row) for row in unitary
            )  # type: ignore[return-value]
    return unitary


def unitary_signature(unitary: Matrix2, digits: int) -> Tuple[float, ...]:
    normalized = remove_global_phase(unitary)
    signature: List[float] = []
    for row in normalized:
        for value in row:
            signature.append(round(float(value.real), digits))
            signature.append(round(float(value.imag), digits))
    return tuple(signature)


def should_skip_extension(sequence: Sequence[str], next_gate: str) -> bool:
    """Prune obvious local redundancies."""
    if not sequence:
        return False

    last_gate = sequence[-1]
    if last_gate == "H" and next_gate == "H":
        return True

    if next_gate == "S" and len(sequence) >= 3 and sequence[-3:] == ["S", "S", "S"]:
        return True

    if next_gate == "T" and len(sequence) >= 7 and sequence[-7:] == ["T"] * 7:
        return True

    return False


@dataclass(order=True)
class SearchNode:
    distance: float
    depth: int
    unitary: Matrix2 = field(compare=False)
    sequence: Tuple[str, ...] = field(compare=False)


@dataclass
class SearchResult:
    target_angle: float
    sequence: Tuple[str, ...]
    unitary: Matrix2
    distance: float
    identity_distance: float
    expansions: int
    threshold_met: bool


@dataclass
class SweepPoint:
    threshold: float
    distance: float
    sequence_length: int
    expansions: int


@dataclass
class MilestoneHit:
    milestone: float
    reached: bool
    distance: float
    sequence: Tuple[str, ...]
    expansions: int


def best_first_search(
    target: Matrix2,
    *,
    tolerance: float,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
) -> SearchResult:
    """Search for a non-empty sequence that beats the identity baseline."""
    identity_distance = challenge_distance(target, IDENTITY)
    frontier: List[SearchNode] = [
        SearchNode(distance=identity_distance, depth=0, unitary=IDENTITY, sequence=())
    ]
    best_depth_by_signature: Dict[Tuple[float, ...], int] = {
        unitary_signature(IDENTITY, merge_digits): 0
    }

    best_sequence: Optional[Tuple[str, ...]] = None
    best_unitary = IDENTITY
    best_distance = identity_distance
    expansions = 0

    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        expansions += 1

        node_is_candidate = bool(node.sequence) and node.distance < identity_distance - 1e-15
        if node_is_candidate and (
            best_sequence is None
            or node.distance < best_distance - 1e-15
            or (
                math.isclose(node.distance, best_distance, abs_tol=1e-15)
                and len(node.sequence) < len(best_sequence)
            )
        ):
            best_sequence = node.sequence
            best_unitary = node.unitary
            best_distance = node.distance

        if node_is_candidate and node.distance <= tolerance:
            return SearchResult(
                target_angle=0.0,
                sequence=node.sequence,
                unitary=node.unitary,
                distance=node.distance,
                identity_distance=identity_distance,
                expansions=expansions,
                threshold_met=True,
            )

        if node.depth >= max_depth:
            continue

        for gate in ("H", "S", "T"):
            if should_skip_extension(node.sequence, gate):
                continue

            child_sequence = node.sequence + (gate,)
            child_unitary = matmul(GATES[gate], node.unitary)
            child_depth = node.depth + 1
            child_signature = unitary_signature(child_unitary, merge_digits)

            previous_best_depth = best_depth_by_signature.get(child_signature)
            if previous_best_depth is not None and child_depth >= previous_best_depth:
                continue

            best_depth_by_signature[child_signature] = child_depth
            child_distance = challenge_distance(target, child_unitary)
            heapq.heappush(
                frontier,
                SearchNode(
                    distance=child_distance,
                    depth=child_depth,
                    unitary=child_unitary,
                    sequence=child_sequence,
                ),
            )

    if best_sequence is None:
        raise RuntimeError(
            "Search did not find any non-empty sequence that beats the identity baseline. "
            "Try increasing --max-depth or --max-expansions."
        )

    return SearchResult(
        target_angle=0.0,
        sequence=best_sequence,
        unitary=best_unitary,
        distance=best_distance,
        identity_distance=identity_distance,
        expansions=expansions,
        threshold_met=best_distance <= tolerance,
    )


def milestone_search(
    target: Matrix2,
    milestones: Sequence[float],
    *,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
) -> List[MilestoneHit]:
    """Track the first sequence that reaches each milestone distance."""
    ordered_milestones = sorted(milestones, reverse=True)
    identity_distance = challenge_distance(target, IDENTITY)
    frontier: List[SearchNode] = [
        SearchNode(distance=identity_distance, depth=0, unitary=IDENTITY, sequence=())
    ]
    best_depth_by_signature: Dict[Tuple[float, ...], int] = {
        unitary_signature(IDENTITY, merge_digits): 0
    }

    hits: Dict[float, MilestoneHit] = {}
    best_sequence: Optional[Tuple[str, ...]] = None
    best_distance = identity_distance
    expansions = 0

    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        expansions += 1

        node_is_candidate = bool(node.sequence) and node.distance < identity_distance - 1e-15
        if node_is_candidate and (
            best_sequence is None
            or node.distance < best_distance - 1e-15
            or (
                math.isclose(node.distance, best_distance, abs_tol=1e-15)
                and len(node.sequence) < len(best_sequence)
            )
        ):
            best_sequence = node.sequence
            best_distance = node.distance

        if node_is_candidate:
            for milestone in ordered_milestones:
                if milestone in hits:
                    continue
                if node.distance <= milestone:
                    hits[milestone] = MilestoneHit(
                        milestone=milestone,
                        reached=True,
                        distance=node.distance,
                        sequence=node.sequence,
                        expansions=expansions,
                    )
            if len(hits) == len(ordered_milestones):
                break

        if node.depth >= max_depth:
            continue

        for gate in ("H", "S", "T"):
            if should_skip_extension(node.sequence, gate):
                continue

            child_sequence = node.sequence + (gate,)
            child_unitary = matmul(GATES[gate], node.unitary)
            child_depth = node.depth + 1
            child_signature = unitary_signature(child_unitary, merge_digits)

            previous_best_depth = best_depth_by_signature.get(child_signature)
            if previous_best_depth is not None and child_depth >= previous_best_depth:
                continue

            best_depth_by_signature[child_signature] = child_depth
            child_distance = challenge_distance(target, child_unitary)
            heapq.heappush(
                frontier,
                SearchNode(
                    distance=child_distance,
                    depth=child_depth,
                    unitary=child_unitary,
                    sequence=child_sequence,
                ),
            )

    fallback_sequence = best_sequence or ()
    fallback_distance = best_distance
    results: List[MilestoneHit] = []
    for milestone in ordered_milestones:
        if milestone in hits:
            results.append(hits[milestone])
        else:
            results.append(
                MilestoneHit(
                    milestone=milestone,
                    reached=False,
                    distance=fallback_distance,
                    sequence=fallback_sequence,
                    expansions=expansions,
                )
            )
    return results


def solve_family(
    ns: Iterable[int],
    *,
    tolerance: float,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for n in ns:
        theta = math.pi / (2**n)
        target = rz(theta)
        result = best_first_search(
            target,
            tolerance=tolerance,
            max_expansions=max_expansions,
            max_depth=max_depth,
            merge_digits=merge_digits,
        )
        result.target_angle = theta
        results.append(result)
    return results


def threshold_sweep(
    n: int,
    thresholds: Sequence[float],
    *,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
) -> List[SweepPoint]:
    theta = math.pi / (2**n)
    target = rz(theta)
    points: List[SweepPoint] = []
    for threshold in thresholds:
        result = best_first_search(
            target,
            tolerance=threshold,
            max_expansions=max_expansions,
            max_depth=max_depth,
            merge_digits=merge_digits,
        )
        points.append(
            SweepPoint(
                threshold=threshold,
                distance=result.distance,
                sequence_length=len(result.sequence),
                expansions=result.expansions,
            )
        )
    return points


def powers_of_ten_thresholds(start_exp: int, end_exp: int) -> List[float]:
    if start_exp > end_exp:
        raise ValueError("start_exp must be <= end_exp")
    return [10.0 ** (-exp) for exp in range(start_exp, end_exp + 1)]


def format_sequence(sequence: Sequence[str]) -> str:
    return " ".join(sequence)


def count_gates(sequence: Sequence[str]) -> Dict[str, int]:
    counts = {"H": 0, "S": 0, "T": 0}
    for gate in sequence:
        counts[gate] += 1
    return counts


def print_results(ns: Sequence[int], results: Sequence[SearchResult]) -> None:
    print("Best-first Clifford+T search for Rz(pi / 2^n)")
    print()

    for n, result in zip(ns, results):
        counts = count_gates(result.sequence)
        print(f"n={n}, target=Rz(pi / 2^{n})")
        print(f"  sequence:          {format_sequence(result.sequence)}")
        print(f"  length:            {len(result.sequence)}")
        print(f"  counts:            H={counts['H']}, S={counts['S']}, T={counts['T']}")
        print(f"  identity_distance: {result.identity_distance:.8f}")
        print(f"  best_distance:     {result.distance:.8f}")
        print(f"  expanded:          {result.expansions}")
        print()


def print_sweep(points: Sequence[SweepPoint], n: int) -> None:
    print(f"Threshold sweep for n={n} (target = Rz(pi / 2^{n}))")
    print()


def print_milestones(points: Sequence[MilestoneHit], n: int) -> None:
    print(f"Milestone search for n={n} (target = Rz(pi / 2^{n}))")
    print()
    print("milestone      reached  distance       length  expansions")
    for point in points:
        print(
            f"{point.milestone:<14.1e}"
            f"{str(point.reached):<9}"
            f"{point.distance:<15.8f}"
            f"{len(point.sequence):<8d}"
            f"{point.expansions}"
        )
    print()
    for point in points:
        print(f"milestone {point.milestone:.0e}")
        print(f"  reached:   {point.reached}")
        print(f"  distance:  {point.distance:.8f}")
        print(f"  length:    {len(point.sequence)}")
        print(f"  expansions:{point.expansions}")
        print(f"  sequence:  {format_sequence(point.sequence) if point.sequence else '<none>'}")
        print()


def maybe_plot_sweep(points: Sequence[SweepPoint], n: int) -> None:
    try:
        import plotext as plt
    except ImportError:
        print("plotext is not installed in this Python environment; skipping plot.")
        return

    x_values = [math.log10(point.threshold) for point in points]
    y_values = [math.log10(point.distance) for point in points]

    plt.clear_data()
    plt.clear_figure()
    plt.plot(x_values, y_values, marker="hd")
    plt.title(f"n={n}: log10(best distance) vs log10(threshold)")
    plt.xlabel("log10(threshold)")
    plt.ylabel("log10(best distance)")
    plt.show()


def maybe_plot_milestones(points: Sequence[MilestoneHit], n: int) -> None:
    try:
        import plotext as plt
    except ImportError:
        print("plotext is not installed in this Python environment; skipping plot.")
        return

    x_values = [math.log10(point.milestone) for point in points]
    y_values = [math.log10(point.distance) for point in points]

    plt.clear_data()
    plt.clear_figure()
    plt.plot(x_values, y_values, marker="hd")
    plt.title(f"n={n}: log10(best distance) vs log10(milestone)")
    plt.xlabel("log10(milestone)")
    plt.ylabel("log10(best distance)")
    plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search for H/S/T sequences that approximate Rz(pi / 2^n)."
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Values of n in the target family Rz(pi / 2^n).",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=3000,
        help="Maximum number of expanded search nodes per target.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=40,
        help="Maximum sequence length to consider.",
    )
    parser.add_argument(
        "--merge-digits",
        type=int,
        default=10,
        help="Rounding precision used when merging nearly equivalent unitaries.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Distance threshold that stops the search early.",
    )
    parser.add_argument(
        "--sweep-n",
        type=int,
        help="Run a threshold sweep for this single n value instead of solving a family.",
    )
    parser.add_argument(
        "--sweep-start-exp",
        type=int,
        default=1,
        help="Start exponent for a powers-of-ten threshold sweep, interpreted as 10^-exp.",
    )
    parser.add_argument(
        "--sweep-end-exp",
        type=int,
        default=7,
        help="End exponent for a powers-of-ten threshold sweep, interpreted as 10^-exp.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot threshold sweep results in the terminal when plotext is available.",
    )
    parser.add_argument(
        "--milestone-n",
        type=int,
        help="Track the first sequence that reaches each powers-of-ten milestone for this n.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.sweep_n is not None:
        thresholds = powers_of_ten_thresholds(args.sweep_start_exp, args.sweep_end_exp)
        points = threshold_sweep(
            args.sweep_n,
            thresholds,
            max_expansions=args.max_expansions,
            max_depth=args.max_depth,
            merge_digits=args.merge_digits,
        )
        print_sweep(points, args.sweep_n)
        if args.plot:
            maybe_plot_sweep(points, args.sweep_n)
        return

    if args.milestone_n is not None:
        milestones = powers_of_ten_thresholds(args.sweep_start_exp, args.sweep_end_exp)
        points = milestone_search(
            rz(math.pi / (2**args.milestone_n)),
            milestones,
            max_expansions=args.max_expansions,
            max_depth=args.max_depth,
            merge_digits=args.merge_digits,
        )
        print_milestones(points, args.milestone_n)
        if args.plot:
            maybe_plot_milestones(points, args.milestone_n)
        return

    results = solve_family(
        args.n_values,
        tolerance=args.tolerance,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        merge_digits=args.merge_digits,
    )
    print_results(args.n_values, results)


if __name__ == "__main__":
    main()
