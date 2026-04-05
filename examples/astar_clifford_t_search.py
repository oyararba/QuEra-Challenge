r"""A*-like search for Clifford+T approximations of dyadic Z rotations.

This script searches over one-qubit gate sequences built from H, S, and T
to approximate the family

    Rz(pi / 2**n),  n in {0, 1, 2, 3, 4, 5}

using the global-phase-invariant distance metric from challenge.md:

    d(U, V) = sqrt(1 - |Tr(U^\dagger V)| / 2)

The search is intentionally "A*-like" instead of strict A*: it uses

    f(sequence) = g(sequence) + heuristic_weight * d(U_target, U_sequence)

where g(sequence) is a weighted path cost with T gates penalized more heavily
than Clifford gates.
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
    """Normalize a unitary by removing an arbitrary global phase."""
    flat = (unitary[0][0], unitary[0][1], unitary[1][0], unitary[1][1])
    for value in flat:
        if abs(value) > eps:
            phase = value / abs(value)
            return tuple(
                tuple(entry / phase for entry in row) for row in unitary
            )  # type: ignore[return-value]
    return unitary


def unitary_signature(unitary: Matrix2, digits: int) -> Tuple[float, ...]:
    """Round a phase-normalized unitary so nearly equivalent states can be merged."""
    normalized = remove_global_phase(unitary)
    signature: List[float] = []
    for row in normalized:
        for value in row:
            signature.append(round(float(value.real), digits))
            signature.append(round(float(value.imag), digits))
    return tuple(signature)


def sequence_cost(
    sequence: Sequence[str],
    t_cost: float,
    clifford_cost: float,
) -> float:
    """Weighted implementation cost for a gate sequence."""
    total = 0.0
    for gate in sequence:
        total += t_cost if gate == "T" else clifford_cost
    return total


def incremental_gate_cost(gate: str, t_cost: float, clifford_cost: float) -> float:
    return t_cost if gate == "T" else clifford_cost


def should_skip_extension(sequence: Sequence[str], next_gate: str) -> bool:
    """Prune a few obvious low-value extensions to keep the search manageable."""
    if not sequence:
        return False

    last_gate = sequence[-1]

    # H H = I, so consecutive H gates never help.
    if last_gate == "H" and next_gate == "H":
        return True

    # S^4 = I, so a fourth consecutive S is redundant.
    if next_gate == "S" and len(sequence) >= 3 and sequence[-3:] == ["S", "S", "S"]:
        return True

    # T^8 = I up to global phase, so very long T runs are wasteful.
    if next_gate == "T" and len(sequence) >= 7 and sequence[-7:] == ["T"] * 7:
        return True

    return False


@dataclass(order=True)
class SearchNode:
    priority: float
    heuristic: float
    g_cost: float
    unitary: Matrix2 = field(compare=False)
    sequence: Tuple[str, ...] = field(compare=False)


@dataclass
class SearchResult:
    target_angle: float
    sequence: Tuple[str, ...]
    unitary: Matrix2
    distance: float
    weighted_cost: float
    expansions: int
    reached_tolerance: bool


def astar_like_search(
    target: Matrix2,
    *,
    t_cost: float,
    clifford_cost: float,
    heuristic_weight: float,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
    tolerance: float,
) -> SearchResult:
    """Search for a low-cost sequence that approximates the target unitary."""
    start: Matrix2 = ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))
    start_distance = challenge_distance(target, start)
    frontier: List[SearchNode] = [
        SearchNode(
            priority=heuristic_weight * start_distance,
            heuristic=start_distance,
            g_cost=0.0,
            unitary=start,
            sequence=(),
        )
    ]
    best_cost_by_signature: Dict[Tuple[float, ...], float] = {
        unitary_signature(start, merge_digits): 0.0
    }

    best_sequence: Tuple[str, ...] = ()
    best_unitary = start
    best_distance = float("inf")
    best_weighted_cost = float("inf")
    found_better_than_identity = False
    expansions = 0

    while frontier and expansions < max_expansions:
        node = heapq.heappop(frontier)
        expansions += 1

        node_distance = challenge_distance(target, node.unitary)
        improves_on_identity = node_distance < start_distance - 1e-15
        if node.sequence and improves_on_identity and (
            (not found_better_than_identity)
            or node_distance < best_distance - 1e-15
            or (
                math.isclose(node_distance, best_distance, abs_tol=1e-15)
                and node.g_cost < best_weighted_cost
            )
        ):
            best_sequence = node.sequence
            best_unitary = node.unitary
            best_distance = node_distance
            best_weighted_cost = node.g_cost
            found_better_than_identity = True

        if node.sequence and improves_on_identity and node_distance <= tolerance:
            return SearchResult(
                target_angle=0.0,
                sequence=node.sequence,
                unitary=node.unitary,
                distance=node_distance,
                weighted_cost=node.g_cost,
                expansions=expansions,
                reached_tolerance=True,
            )

        if len(node.sequence) >= max_depth:
            continue

        for gate in ("H", "S", "T"):
            if should_skip_extension(node.sequence, gate):
                continue

            child_sequence = node.sequence + (gate,)
            child_unitary = matmul(GATES[gate], node.unitary)
            child_g = node.g_cost + incremental_gate_cost(gate, t_cost, clifford_cost)
            child_distance = challenge_distance(target, child_unitary)
            child_signature = unitary_signature(child_unitary, merge_digits)

            previous_best = best_cost_by_signature.get(child_signature)
            if previous_best is not None and child_g >= previous_best - 1e-12:
                continue

            best_cost_by_signature[child_signature] = child_g
            child_priority = child_g + heuristic_weight * child_distance
            heapq.heappush(
                frontier,
                SearchNode(
                    priority=child_priority,
                    heuristic=child_distance,
                    g_cost=child_g,
                    unitary=child_unitary,
                    sequence=child_sequence,
                ),
            )

    if not found_better_than_identity:
        raise RuntimeError(
            "Search did not find any non-empty sequence that beats the identity baseline. "
            "Try increasing --max-depth or --max-expansions, or lowering --t-cost."
        )

    return SearchResult(
        target_angle=0.0,
        sequence=best_sequence,
        unitary=best_unitary,
        distance=best_distance,
        weighted_cost=best_weighted_cost,
        expansions=expansions,
        reached_tolerance=best_distance <= tolerance,
    )


def format_sequence(sequence: Sequence[str]) -> str:
    return " ".join(sequence) if sequence else "<no-sequence>"


def count_gates(sequence: Sequence[str]) -> Dict[str, int]:
    counts = {"H": 0, "S": 0, "T": 0}
    for gate in sequence:
        counts[gate] += 1
    return counts


def solve_family(
    ns: Iterable[int],
    *,
    t_cost: float,
    clifford_cost: float,
    heuristic_weight: float,
    max_expansions: int,
    max_depth: int,
    merge_digits: int,
    tolerance: float,
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for n in ns:
        theta = math.pi / (2**n)
        target = rz(theta)
        result = astar_like_search(
            target,
            t_cost=t_cost,
            clifford_cost=clifford_cost,
            heuristic_weight=heuristic_weight,
            max_expansions=max_expansions,
            max_depth=max_depth,
            merge_digits=merge_digits,
            tolerance=tolerance,
        )
        result.target_angle = theta
        results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Approximate Rz(pi / 2^n) using an A*-like Clifford+T search."
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Values of n in the target family Rz(pi / 2^n).",
    )
    parser.add_argument(
        "--t-cost",
        type=float,
        default=3.0,
        help="Path-cost weight for each T gate.",
    )
    parser.add_argument(
        "--clifford-cost",
        type=float,
        default=1.0,
        help="Path-cost weight for each Clifford gate (H or S).",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=40.0,
        help="Multiplier on the challenge distance in the A*-like priority.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=120000,
        help="Maximum number of expanded search nodes per target.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=28,
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
        default=0.01,
        help="Distance threshold treated as a successful approximation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    results = solve_family(
        args.n_values,
        t_cost=args.t_cost,
        clifford_cost=args.clifford_cost,
        heuristic_weight=args.heuristic_weight,
        max_expansions=args.max_expansions,
        max_depth=args.max_depth,
        merge_digits=args.merge_digits,
        tolerance=args.tolerance,
    )

    print("A*-like Clifford+T search for Rz(pi / 2^n)")
    print(
        f"weights: T={args.t_cost:g}, Clifford={args.clifford_cost:g}, "
        f"heuristic_weight={args.heuristic_weight:g}"
    )
    print(
        f"limits: max_depth={args.max_depth}, max_expansions={args.max_expansions}, "
        f"tolerance={args.tolerance:g}"
    )
    print()

    for n, result in zip(args.n_values, results):
        counts = count_gates(result.sequence)
        print(f"n={n}, target=Rz(pi / 2^{n})")
        print(f"  sequence: {format_sequence(result.sequence)}")
        print(f"  length:   {len(result.sequence)}")
        print(
            f"  counts:   H={counts['H']}, S={counts['S']}, T={counts['T']}"
        )
        print(f"  cost:     {result.weighted_cost:.3f}")
        print(f"  distance: {result.distance:.8f}")
        print(f"  expanded: {result.expansions}")
        print()


if __name__ == "__main__":
    main()
