"""Spatial cache benchmark: python task1p.py

Dependencies: numpy, numba, psutil (pip install numpy numba psutil).
Uses a dense C-contiguous 8000 x 8000 float64 matrix and native Numba threads.
"""

import math
import os
import platform
import statistics
import time

import numba
import numpy as np
import psutil
from numba import njit, prange

STUDENT_ID = "230103254"
SIDE = 8_000
REPEATS = 5


@njit(parallel=True)
def row_major(matrix, partials):
    rows, columns = matrix.shape
    workers = len(partials)
    for worker in prange(workers):
        total = 0.0
        begin = rows * worker // workers
        end = rows * (worker + 1) // workers
        for row in range(begin, end):
            for column in range(columns):
                total += matrix[row, column]
        partials[worker] = total


@njit(parallel=True)
def column_major(matrix, partials):
    rows, columns = matrix.shape
    workers = len(partials)
    for worker in prange(workers):
        total = 0.0
        begin = columns * worker // workers
        end = columns * (worker + 1) // workers
        for column in range(begin, end):
            for row in range(rows):
                total += matrix[row, column]
        partials[worker] = total


def measure(kernel, matrix, partials):
    start = time.perf_counter_ns()
    kernel(matrix, partials)  # Returns after all native workers finish.
    total = math.fsum(partials)
    elapsed_ns = time.perf_counter_ns() - start
    return elapsed_ns / 1_000_000, total


def check_sum(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-5):
        raise AssertionError(f"Sum mismatch: {actual:.17g} vs {expected:.17g}")


def main():
    physical = psutil.cpu_count(logical=False)
    if physical is None or physical < 1:
        raise RuntimeError("Could not detect physical core count")
    numba.set_num_threads(physical)
    print(f"Student ID / seed: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / platform: {platform.processor() or platform.machine()}")
    print(f"Logical core count (os.cpu_count): {os.cpu_count()}")
    print(f"Physical core count (psutil): {physical}")
    print(f"Native worker threads (Numba): {numba.get_num_threads()}")
    print(f"Versions: Python {platform.python_version()}, NumPy {np.__version__}, Numba {numba.__version__}")
    print(f"Matrix: {SIDE:,} x {SIDE:,}, float64, C-contiguous (row-major)")
    print(f"Matrix storage: {SIDE * SIDE * 8:,} bytes")
    print("Input: numpy.random.default_rng(Student ID), uniform [0, 1).")
    print("Mode A: contiguous row blocks; inner loop traverses columns (8-byte stride).")
    print(f"Mode B: contiguous column blocks; inner loop traverses rows ({SIDE * 8:,}-byte stride).")
    print("Threads are scheduled by the OS; no explicit physical-core affinity is imposed.")
    print("Timing includes dispatch, all worker loops, and the final partial-sum reduction.")
    print("Allocation, generation, JIT compilation, warm-up, and validation are excluded.")
    print("Five measured runs per mode, alternating A/B order; headline times are medians.")

    partials = np.empty(physical, dtype=np.float64)
    # Compile and initialize the worker runtime before allocating the full matrix.
    small = np.ones((64, 64), dtype=np.float64)
    for kernel in (row_major, column_major):
        _, total = measure(kernel, small, partials)
        check_sum(total, small.size)
    print(f"Numba threading backend: {numba.threading_layer()}", flush=True)

    matrix = np.random.default_rng(int(STUDENT_ID)).random((SIDE, SIDE), dtype=np.float64)
    assert matrix.flags.c_contiguous and matrix.dtype == np.float64
    reference = float(np.sum(matrix, dtype=np.float64))
    for kernel in (row_major, column_major):
        _, total = measure(kernel, matrix, partials)  # One full-matrix warm-up per mode.
        check_sum(total, reference)

    samples = {"A": [], "B": []}
    sums = {}
    for repeat in range(REPEATS):
        order = [("A", row_major), ("B", column_major)]
        if repeat % 2:
            order.reverse()
        for label, kernel in order:
            elapsed_ms, total = measure(kernel, matrix, partials)
            check_sum(total, reference)
            samples[label].append(elapsed_ms)
            sums[label] = total

    a_ms = statistics.median(samples["A"])
    b_ms = statistics.median(samples["B"])
    print(f"\nReference sum: {reference:.17g}")
    for label in ("A", "B"):
        print(f"Mode {label} sum: {sums[label]:.17g}")
        print(f"Mode {label} raw wall times (ms): " + ", ".join(f"{t:.6f}" for t in samples[label]))
    print(f"\nMode A (Row-Major) runtime, median (ms): {a_ms:.6f}")
    print(f"Mode B (Column-Major / Strided) runtime, median (ms): {b_ms:.6f}")
    print(f"Cache penalty factor (Mode B Time / Mode A Time): {b_ms / a_ms:.4f}x")
    print("Validation passed: both traversal sums match the reference within floating-point tolerance.")
    print("The factor measures traversal-time differences; hardware cache misses were not directly counted.")


if __name__ == "__main__":
    main()
