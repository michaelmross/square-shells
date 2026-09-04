#!/usr/bin/env python3
"""
overnight_shifted_hierarchy_v3.py

Dense scan of square shells for the shifted Bonferroni / multiplication
coverage inequalities, with checkpoint/resume support.

For odd K and integer r >= 2 define

    P_{K,r}(w)
      = 1 + (w-1) * product_{j=0}^{K-2}(w-r-j)
                    / product_{j=0}^{K-2}(r+j).

For every integer w >= 1, P_{K,r}(w) >= 1, while P_{K,r}(0)=0.
Therefore, if w(m) is the number of irreducible odd multiplication rows
hitting an odd shell position m,

    covered <= sum_m P_{K,r}(w(m)).

A hole is certified whenever this upper bound is < n, since there are
exactly n odd positions in (n^2,(n+1)^2).

Outputs
-------
1. Main CSV:
   regular sampled rows plus first failures and best-r transitions.

2. Transitions CSV:
   initial best-r choices, every best-r change, and periodic heartbeat rows.

3. Checkpoint JSON:
   exact last completed n and complete tracker state.  It is written
   atomically and is what makes interruption/resume exact.

Normal run
----------
    python overnight_shifted_hierarchy_v3.py ^
        --start 10001 --stop 1000000 ^
        --orders 3,5,7,9,11 ^
        --rmax 12 ^
        --workers 2 ^
        --write-every 1000 ^
        --checkpoint-every 1000 ^
        --heartbeat-every 10000 ^
        --out overnight_hierarchy.csv

Resume after Ctrl-C / shutdown
------------------------------
    python overnight_shifted_hierarchy_v3.py ^
        --resume ^
        --stop 1000000 ^
        --orders 3,5,7,9,11 ^
        --rmax 12 ^
        --workers 2 ^
        --out overnight_hierarchy.csv

If a v3 checkpoint exists, resume is exact from last_completed_n + 1.

Legacy resume
-------------
If --resume is used on files made by v2 and no checkpoint exists yet, v3
bootstraps safely from the largest n actually stored in the main CSV.  With
--write-every 1000, this can re-do at most about 999 shells.  Current best-r
states and transition counts are reconstructed from the transitions CSV.

Because v2 did not persist every shell's cumulative failure counter/worst
ratio, those particular historical aggregates cannot be reconstructed
exactly without re-running the old range.  v3 labels that limitation in its
summary.  From the first v3 checkpoint onward, interruption/resume is exact.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROWS: list[int] = []
ORDERS: tuple[int, ...] = ()
RMAX: int = 12
CHECKPOINT_VERSION = 1


def irreducible_odd_rows(limit: int) -> list[int]:
    """Construct odd irreducible multiplication rows <= limit by marking."""
    if limit < 3:
        return []
    reducible = bytearray(limit + 1)
    rows: list[int] = []
    for a in range(3, limit + 1, 2):
        if reducible[a]:
            continue
        rows.append(a)
        aa = a * a
        if aa <= limit:
            step = 2 * a
            count = ((limit - aa) // step) + 1
            reducible[aa : limit + 1 : step] = b"\x01" * count
    return rows


def init_worker(stop: int, orders: tuple[int, ...], rmax: int) -> None:
    global ROWS, ORDERS, RMAX
    ROWS = irreducible_odd_rows(stop)
    ORDERS = orders
    RMAX = rmax


def first_odd_in_shell(n: int) -> int:
    return n * n + (1 if n % 2 == 0 else 2)


def depth_histogram(n: int) -> Counter[int]:
    """
    Histogram of w(m), the number of irreducible odd rows a<=n dividing m,
    over the n odd positions in n^2 < m < (n+1)^2.
    """
    depths = bytearray(n)
    lo = n * n
    hi = (n + 1) * (n + 1)
    lo_odd = first_odd_in_shell(n)

    upto = bisect.bisect_right(ROWS, n)
    for a in ROWS[:upto]:
        b = lo // a + 1
        if b % 2 == 0:
            b += 1
        m0 = a * b
        step = 2 * a
        for m in range(m0, hi, step):
            depths[(m - lo_odd) // 2] += 1

    return Counter(depths)


def denominator(K: int, r: int) -> int:
    d = 1
    for j in range(K - 1):
        d *= r + j
    return d


def scaled_bound_and_margin(
    hist: Counter[int], n: int, K: int, r: int
) -> tuple[int, int, int]:
    """
    Return (bound_numerator, denominator, margin_numerator), exactly.
    """
    D = denominator(K, r)
    B = 0
    for w, count in hist.items():
        if w == 0:
            continue

        prod = w - 1
        for j in range(K - 1):
            prod *= w - r - j

        term = D + prod
        if term < D:
            raise AssertionError(
                f"pointwise majorant failed: K={K}, r={r}, w={w}"
            )
        B += count * term

    M = n * D - B
    return B, D, M


def analyze_n(n: int) -> tuple:
    hist = depth_histogram(n)
    exact_holes = hist.get(0, 0)
    max_depth = max(hist) if hist else 0

    results = []
    for K in ORDERS:
        B3, D3, M3 = scaled_bound_and_margin(hist, n, K, 3)
        fixed_ratio = B3 / (D3 * n)
        fixed_margin = M3 / D3

        B2, D2, M2 = scaled_bound_and_margin(hist, n, K, 2)
        bonf_ratio = B2 / (D2 * n)
        bonf_margin = M2 / D2

        best_r = 2
        best_margin_num = M2
        best_D = D2
        best_ratio = bonf_ratio

        for r in range(3, RMAX + 1):
            B, D, M = scaled_bound_and_margin(hist, n, K, r)
            ratio = B / (D * n)

            # Compare exact margins M/D by cross multiplication.
            if M * best_D > best_margin_num * D:
                best_r = r
                best_margin_num = M
                best_D = D
                best_ratio = ratio

        best_margin = best_margin_num / best_D

        results.append(
            (
                K,
                fixed_ratio,
                fixed_margin,
                bonf_ratio,
                bonf_margin,
                best_r,
                best_ratio,
                best_margin,
            )
        )

    return n, exact_holes, max_depth, tuple(results)


@dataclass
class Tracker:
    first_fixed_failure: int | None = None
    fixed_failures: int = 0
    worst_fixed_ratio: float = -1.0
    worst_fixed_n: int | None = None

    first_best_failure: int | None = None
    best_failures: int = 0
    worst_best_ratio: float = -1.0
    worst_best_n: int | None = None
    worst_best_r: int | None = None

    max_best_r: int = 0
    best_r_changes: int = 0
    previous_best_r: int | None = None


def parse_orders(s: str) -> tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if not vals:
        raise argparse.ArgumentTypeError("empty order list")
    if any(k < 1 or k % 2 == 0 for k in vals):
        raise argparse.ArgumentTypeError("orders must be positive odd integers")
    return vals


def main_fieldnames(orders: tuple[int, ...]) -> list[str]:
    fields = ["n", "exact_holes", "max_depth"]
    for K in orders:
        fields += [
            f"K{K}_r3_ratio",
            f"K{K}_r3_margin",
            f"K{K}_bonf_ratio",
            f"K{K}_bonf_margin",
            f"K{K}_best_r",
            f"K{K}_best_ratio",
            f"K{K}_best_margin",
        ]
    return fields


TRANSITION_FIELDS = [
    "n",
    "K",
    "event",
    "old_best_r",
    "new_best_r",
    "best_ratio",
    "best_margin",
    "exact_holes",
    "max_depth",
]


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    path: Path,
    *,
    scan_start: int,
    last_completed_n: int,
    stop: int,
    orders: tuple[int, ...],
    rmax: int,
    trackers: dict[int, Tracker],
    history_complete: bool,
    legacy_resume_n: int | None,
    completed: bool,
) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "scan_start": scan_start,
        "last_completed_n": last_completed_n,
        "stop_requested": stop,
        "orders": list(orders),
        "rmax": rmax,
        "history_complete": history_complete,
        "legacy_resume_n": legacy_resume_n,
        "completed": completed,
        "trackers": {str(K): asdict(trackers[K]) for K in orders},
        "saved_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(path, payload)


def load_checkpoint(
    path: Path, orders: tuple[int, ...], rmax: int
) -> tuple[int, int, dict[int, Tracker], bool, int | None]:
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    if state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise SystemExit(
            f"unsupported checkpoint version in {path}: "
            f"{state.get('checkpoint_version')}"
        )
    if tuple(state.get("orders", [])) != orders:
        raise SystemExit(
            f"checkpoint orders {state.get('orders')} do not match "
            f"command line {list(orders)}"
        )
    if int(state.get("rmax")) != rmax:
        raise SystemExit(
            f"checkpoint rmax={state.get('rmax')} does not match --rmax {rmax}"
        )

    trackers: dict[int, Tracker] = {}
    raw_trackers = state.get("trackers", {})
    for K in orders:
        if str(K) not in raw_trackers:
            raise SystemExit(f"checkpoint is missing tracker K={K}")
        trackers[K] = Tracker(**raw_trackers[str(K)])

    return (
        int(state["scan_start"]),
        int(state["last_completed_n"]),
        trackers,
        bool(state.get("history_complete", True)),
        state.get("legacy_resume_n"),
    )


def read_main_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bootstrap_legacy_resume(
    main_path: Path,
    transitions_path: Path,
    orders: tuple[int, ...],
) -> tuple[int, int, dict[int, Tracker]]:
    """
    Bootstrap from a v2 sampled/event CSV when no v3 checkpoint exists.

    Returns:
        scan_start, last_safe_n, trackers

    The last stored main-CSV n is a conservative safe restart point.
    """
    if not main_path.exists():
        raise SystemExit(
            f"--resume requested but neither checkpoint nor main CSV exists: "
            f"{main_path}"
        )

    rows = read_main_rows(main_path)
    if not rows:
        raise SystemExit(f"cannot resume: {main_path} contains no data rows")

    try:
        ns = [int(row["n"]) for row in rows]
    except Exception as exc:
        raise SystemExit(f"cannot read n column from {main_path}: {exc}")

    scan_start = min(ns)
    last_safe_n = max(ns)
    trackers = {K: Tracker() for K in orders}

    # Recover what can be known from the sampled/event main CSV.
    # First-failure rows were forced into v2 output, so first failures are exact.
    # Worst ratios below are only worst among persisted historical rows.
    for row in rows:
        n = int(row["n"])
        if n > last_safe_n:
            continue
        for K in orders:
            tr = trackers[K]

            fixed_key = f"K{K}_r3_ratio"
            best_key = f"K{K}_best_ratio"
            best_r_key = f"K{K}_best_r"

            if fixed_key in row and row[fixed_key]:
                x = float(row[fixed_key])
                if x >= 1.0 and (
                    tr.first_fixed_failure is None or n < tr.first_fixed_failure
                ):
                    tr.first_fixed_failure = n
                if x > tr.worst_fixed_ratio:
                    tr.worst_fixed_ratio = x
                    tr.worst_fixed_n = n

            if best_key in row and row[best_key]:
                x = float(row[best_key])
                r = int(row[best_r_key])
                if x >= 1.0 and (
                    tr.first_best_failure is None or n < tr.first_best_failure
                ):
                    tr.first_best_failure = n
                if x > tr.worst_best_ratio:
                    tr.worst_best_ratio = x
                    tr.worst_best_n = n
                    tr.worst_best_r = r
                tr.max_best_r = max(tr.max_best_r, r)

    # Recover exact current best-r states and exact transition counts from
    # the compact transition log, if available.
    if transitions_path.exists():
        with transitions_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    n = int(row["n"])
                    K = int(row["K"])
                except (ValueError, KeyError):
                    continue
                if n > last_safe_n or K not in trackers:
                    continue

                tr = trackers[K]
                event = row.get("event", "")
                new_r_text = row.get("new_best_r", "")
                if new_r_text:
                    new_r = int(new_r_text)
                    tr.previous_best_r = new_r
                    tr.max_best_r = max(tr.max_best_r, new_r)
                if event == "best_r_change":
                    tr.best_r_changes += 1

    # Fallback if there was no usable transitions file: use the last stored
    # main row at last_safe_n.
    last_rows = [row for row in rows if int(row["n"]) == last_safe_n]
    if last_rows:
        row = last_rows[-1]
        for K in orders:
            tr = trackers[K]
            if tr.previous_best_r is None:
                key = f"K{K}_best_r"
                if key in row and row[key]:
                    tr.previous_best_r = int(row[key])
                    tr.max_best_r = max(tr.max_best_r, tr.previous_best_r)

    # v2 did not persist exact cumulative counts for all shells.
    tr_missing = [K for K, tr in trackers.items() if tr.previous_best_r is None]
    if tr_missing:
        raise SystemExit(
            "cannot reconstruct current best-r state for K="
            + ",".join(map(str, tr_missing))
        )

    return scan_start, last_safe_n, trackers


def open_csv_writer(
    path: Path, fieldnames: list[str], append: bool
) -> tuple[object, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists_nonempty = path.exists() and path.stat().st_size > 0

    mode = "a" if append else "w"
    f = path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    if not append or not exists_nonempty:
        writer.writeheader()
        f.flush()

    return f, writer


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dense shifted-Bonferroni hierarchy scan with resume."
    )
    ap.add_argument("--start", type=int, default=10001)
    ap.add_argument("--stop", type=int, default=1000000)
    ap.add_argument("--orders", type=parse_orders, default=(3, 5, 7, 9, 11))
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = all but one CPU core; 1 = serial",
    )
    ap.add_argument("--chunksize", type=int, default=64)
    ap.add_argument(
        "--write-every",
        type=int,
        default=1000,
        help="sample one main-CSV row every this many n",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="print progress every this many shells in the current session",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="save exact resume state every this many completed shells",
    )
    ap.add_argument(
        "--heartbeat-every",
        type=int,
        default=10000,
        help="write current best-r states to transitions CSV this often; 0 disables",
    )
    ap.add_argument("--out", type=Path, default=Path("overnight_hierarchy.csv"))
    ap.add_argument(
        "--transitions-out",
        type=Path,
        default=None,
        help="default: <out stem>_transitions.csv",
    )
    ap.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="default: <out stem>.checkpoint.json",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="append and continue from checkpoint; bootstrap safely from v2 if needed",
    )
    ap.add_argument(
        "--print-transitions",
        action="store_true",
        help="also print every best-r change",
    )
    args = ap.parse_args()

    if args.start < 2:
        raise SystemExit("--start must be >= 2")
    if args.stop < 2:
        raise SystemExit("--stop must be >= 2")
    if not args.resume and args.stop < args.start:
        raise SystemExit("require start <= stop")
    if args.rmax < 3:
        raise SystemExit("--rmax must be at least 3")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be >= 1")
    if args.heartbeat_every < 0:
        raise SystemExit("--heartbeat-every must be >= 0")

    if args.transitions_out is None:
        args.transitions_out = args.out.with_name(
            args.out.stem + "_transitions.csv"
        )
    if args.checkpoint_out is None:
        args.checkpoint_out = args.out.with_name(
            args.out.stem + ".checkpoint.json"
        )

    cpu = os.cpu_count() or 2
    workers = args.workers if args.workers > 0 else max(1, cpu - 1)

    history_complete = True
    legacy_resume_n: int | None = None

    if args.resume:
        if args.checkpoint_out.exists():
            (
                scan_start,
                last_completed_n,
                trackers,
                history_complete,
                legacy_resume_n,
            ) = load_checkpoint(args.checkpoint_out, args.orders, args.rmax)
            resume_source = f"checkpoint {args.checkpoint_out}"
        else:
            (
                scan_start,
                last_completed_n,
                trackers,
            ) = bootstrap_legacy_resume(
                args.out, args.transitions_out, args.orders
            )
            history_complete = False
            legacy_resume_n = last_completed_n
            resume_source = "legacy v2 CSV bootstrap"

        effective_start = last_completed_n + 1

        if args.stop < last_completed_n:
            raise SystemExit(
                f"--stop {args.stop} is below checkpointed n={last_completed_n}"
            )

        if effective_start > args.stop:
            print(
                f"Nothing to do: already completed through n={last_completed_n:,}; "
                f"requested stop is {args.stop:,}."
            )
            return
    else:
        scan_start = args.start
        last_completed_n = args.start - 1
        effective_start = args.start
        trackers = {K: Tracker() for K in args.orders}
        resume_source = None

    main_fields = main_fieldnames(args.orders)

    print("Dense shifted-hierarchy scan (v3)")
    if args.resume:
        print(f"  RESUME: {resume_source}")
        print(f"  original scan start: {scan_start:,}")
        print(f"  safely completed through: {last_completed_n:,}")
        print(f"  continuing at: {effective_start:,}")
        if not history_complete:
            print(
                "  NOTE: v2 historical failure counts/worst ratios were not "
                "fully persisted; those summary aggregates will be labelled partial."
            )
    else:
        print(f"  n: {effective_start:,} .. {args.stop:,}  (EVERY shell)")

    print(f"  requested stop: {args.stop:,}")
    print(f"  orders: {','.join(map(str, args.orders))}")
    print(f"  shifted roots: r=2..{args.rmax}  (r=2 is Bonferroni)")
    print(f"  workers: {workers}")
    print(f"  main CSV: {args.out}")
    print(f"  transitions CSV: {args.transitions_out}")
    print(f"  checkpoint: {args.checkpoint_out}")
    print(
        f"  checkpoint every {args.checkpoint_every:,} shells; "
        f"heartbeat every {args.heartbeat_every:,} shells"
        if args.heartbeat_every
        else f"  checkpoint every {args.checkpoint_every:,} shells; heartbeat disabled"
    )
    print()

    f, writer = open_csv_writer(args.out, main_fields, append=args.resume)
    tf, transition_writer = open_csv_writer(
        args.transitions_out, TRANSITION_FIELDS, append=args.resume
    )

    t0 = time.time()
    session_done = 0
    interrupted = False

    def checkpoint(completed: bool = False) -> None:
        f.flush()
        tf.flush()
        save_checkpoint(
            args.checkpoint_out,
            scan_start=scan_start,
            last_completed_n=last_completed_n,
            stop=args.stop,
            orders=args.orders,
            rmax=args.rmax,
            trackers=trackers,
            history_complete=history_complete,
            legacy_resume_n=legacy_resume_n,
            completed=completed,
        )

    def consume(records: Iterable[tuple]) -> None:
        nonlocal last_completed_n, session_done

        for rec in records:
            n, exact_holes, max_depth, kres = rec
            row = {
                "n": n,
                "exact_holes": exact_holes,
                "max_depth": max_depth,
            }
            write_this = (
                n == effective_start
                or n == args.stop
                or (args.write_every > 0 and n % args.write_every == 0)
            )

            heartbeat = bool(
                args.heartbeat_every > 0 and n % args.heartbeat_every == 0
            )

            for (
                K,
                fixed_ratio,
                fixed_margin,
                bonf_ratio,
                bonf_margin,
                best_r,
                best_ratio,
                best_margin,
            ) in kres:
                tr = trackers[K]

                if tr.previous_best_r is None:
                    tr.previous_best_r = best_r
                    transition_writer.writerow(
                        {
                            "n": n,
                            "K": K,
                            "event": "initial_best_r",
                            "old_best_r": "",
                            "new_best_r": best_r,
                            "best_ratio": f"{best_ratio:.12g}",
                            "best_margin": f"{best_margin:.12g}",
                            "exact_holes": exact_holes,
                            "max_depth": max_depth,
                        }
                    )
                    write_this = True

                elif best_r != tr.previous_best_r:
                    old_best_r = tr.previous_best_r
                    tr.previous_best_r = best_r
                    tr.best_r_changes += 1

                    transition_writer.writerow(
                        {
                            "n": n,
                            "K": K,
                            "event": "best_r_change",
                            "old_best_r": old_best_r,
                            "new_best_r": best_r,
                            "best_ratio": f"{best_ratio:.12g}",
                            "best_margin": f"{best_margin:.12g}",
                            "exact_holes": exact_holes,
                            "max_depth": max_depth,
                        }
                    )
                    write_this = True

                    if args.print_transitions:
                        print(
                            f"BEST-r CHANGE: K={K}, n={n:,}, "
                            f"{old_best_r} -> {best_r}, "
                            f"ratio={best_ratio:.9f}"
                        )

                if fixed_ratio >= 1.0:
                    tr.fixed_failures += 1
                    if tr.first_fixed_failure is None:
                        tr.first_fixed_failure = n
                        write_this = True
                        print(
                            f"FIRST r=3 FAILURE: K={K}, "
                            f"n={n:,}, ratio={fixed_ratio:.9f}"
                        )

                if fixed_ratio > tr.worst_fixed_ratio:
                    tr.worst_fixed_ratio = fixed_ratio
                    tr.worst_fixed_n = n

                if best_ratio >= 1.0:
                    tr.best_failures += 1
                    if tr.first_best_failure is None:
                        tr.first_best_failure = n
                        write_this = True
                        print(
                            f"FIRST best-r FAILURE: K={K}, "
                            f"n={n:,}, best r={best_r}, "
                            f"ratio={best_ratio:.9f}"
                        )

                if best_ratio > tr.worst_best_ratio:
                    tr.worst_best_ratio = best_ratio
                    tr.worst_best_n = n
                    tr.worst_best_r = best_r

                tr.max_best_r = max(tr.max_best_r, best_r)

                row.update(
                    {
                        f"K{K}_r3_ratio": f"{fixed_ratio:.12g}",
                        f"K{K}_r3_margin": f"{fixed_margin:.12g}",
                        f"K{K}_bonf_ratio": f"{bonf_ratio:.12g}",
                        f"K{K}_bonf_margin": f"{bonf_margin:.12g}",
                        f"K{K}_best_r": best_r,
                        f"K{K}_best_ratio": f"{best_ratio:.12g}",
                        f"K{K}_best_margin": f"{best_margin:.12g}",
                    }
                )

                if heartbeat:
                    transition_writer.writerow(
                        {
                            "n": n,
                            "K": K,
                            "event": "heartbeat",
                            "old_best_r": best_r,
                            "new_best_r": best_r,
                            "best_ratio": f"{best_ratio:.12g}",
                            "best_margin": f"{best_margin:.12g}",
                            "exact_holes": exact_holes,
                            "max_depth": max_depth,
                        }
                    )

            if write_this:
                writer.writerow(row)

            # State is valid through n only after all trackers / output rows above.
            last_completed_n = n
            session_done += 1

            if session_done % args.checkpoint_every == 0:
                checkpoint(completed=False)

            if args.progress_every and session_done % args.progress_every == 0:
                elapsed = time.time() - t0
                rate = session_done / elapsed if elapsed else 0.0
                remaining = args.stop - n
                eta = remaining / rate if rate else float("inf")
                print(
                    f"progress: session {session_done:,} shells, "
                    f"completed n={n:,}; "
                    f"{rate:.2f} shells/s; "
                    f"ETA {eta/3600:.2f} h"
                )

    executor: ProcessPoolExecutor | None = None
    try:
        if workers == 1:
            init_worker(args.stop, args.orders, args.rmax)
            consume(analyze_n(n) for n in range(effective_start, args.stop + 1))
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(args.stop, args.orders, args.rmax),
            )
            records = executor.map(
                analyze_n,
                range(effective_start, args.stop + 1),
                chunksize=args.chunksize,
            )
            consume(records)

    except KeyboardInterrupt:
        interrupted = True
        print()
        print("Ctrl-C received. Saving checkpoint...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None

    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        checkpoint(completed=(not interrupted and last_completed_n >= args.stop))
        f.close()
        tf.close()

    elapsed = time.time() - t0

    print()
    print("=" * 76)
    print("INTERRUPTED SUMMARY" if interrupted else "FINAL SUMMARY")
    print("=" * 76)
    print(
        f"This session processed {session_done:,} shells "
        f"through n={last_completed_n:,} in {elapsed/3600:.2f} h."
    )
    print(f"Resume checkpoint: {args.checkpoint_out}")
    if interrupted:
        print(
            f"Next --resume run will continue at n={last_completed_n + 1:,}."
        )
    print()

    if history_complete:
        denominator_count = max(0, last_completed_n - scan_start + 1)
        failure_scope = f"/{denominator_count:,}"
        worst_note = ""
    else:
        resumed_count = max(0, last_completed_n - (legacy_resume_n or last_completed_n))
        failure_scope = f"/{resumed_count:,} since legacy resume"
        worst_note = " (old portion = persisted rows only)"

    for K in args.orders:
        tr = trackers[K]
        print(f"K={K}")
        print(
            "  fixed r=3: "
            f"first failure = "
            f"{tr.first_fixed_failure if tr.first_fixed_failure is not None else 'NONE'}, "
            f"failures = {tr.fixed_failures:,}{failure_scope}, "
            f"worst ratio = {tr.worst_fixed_ratio:.9f} "
            f"at n={tr.worst_fixed_n}{worst_note}"
        )
        print(
            f"  best r in 2..{args.rmax}: "
            f"first failure = "
            f"{tr.first_best_failure if tr.first_best_failure is not None else 'NONE'}, "
            f"failures = {tr.best_failures:,}{failure_scope}, "
            f"worst ratio = {tr.worst_best_ratio:.9f} "
            f"at n={tr.worst_best_n} "
            f"(best r there={tr.worst_best_r}){worst_note}"
        )
        print(f"  current best r: {tr.previous_best_r}")
        print(f"  largest best-r selected: {tr.max_best_r}")
        print(f"  number of best-r changes recorded: {tr.best_r_changes:,}")
        print()

    print(f"Main sampled/event detail: {args.out}")
    print(f"Transitions + heartbeat log: {args.transitions_out}")
    print(f"Exact resume state: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
