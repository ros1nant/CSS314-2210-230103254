"""Process-based Amdahl benchmark. Run: python task2.py (standard library only)."""

from array import array
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import random
import time

STUDENT_ID = "230103254"
ELEMENT_COUNT = 20_000_000
PARALLEL_FRACTION = 0.95


def chunk_sum(task):
    """Sum a disjoint float64 slice without copying the shared input array."""
    name, start, stop = task
    shared = shared_memory.SharedMemory(name=name)
    values = shared.buf.cast("d")
    chunk = values[start:stop]
    try:
        return math.fsum(chunk)
    finally:
        chunk.release()
        values.release()
        shared.close()


def benchmark(context, name, size, workers):
    start_ns = time.perf_counter_ns()
    tasks = [
        (name, size * i // workers, size * (i + 1) // workers)
        for i in range(workers)
    ]
    with context.Pool(processes=workers) as pool:
        total = math.fsum(pool.map(chunk_sum, tasks, chunksize=1))
    elapsed_ns = time.perf_counter_ns() - start_ns
    return total, elapsed_ns


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = list(dict.fromkeys([1, 2, 4, max_cores, max_cores * 2]))
    print(f"Student ID: {STUDENT_ID}", flush=True)
    print(f"OS: {platform.platform()}")
    print(f"CPU / platform details: {platform.processor() or platform.machine()}")
    print(f"Total logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Array: {ELEMENT_COUNT:,} float64 elements, uniform [0, 1)")
    print(f"Seed: {int(STUDENT_ID)}")
    print(f"Worker counts: {worker_counts}")
    print("Cores column denotes worker processes; each worker sums serially.")
    print("One measured run per count, using the spawn start method.")
    print("Timing includes process startup, dispatch, summation, reduction, and shutdown.")
    print("Array generation, shared-memory setup, and validation are excluded.")
    print("Speedup = T(1) / T(p), using the one-worker summation baseline.")
    print("Amdahl upper bound = 1 / (0.05 + 0.95 / p); infinite-worker limit = 20x.")

    rng = random.Random(int(STUDENT_ID))
    values = array("d", (rng.random() for _ in range(ELEMENT_COUNT)))
    if values.itemsize != 8:
        raise RuntimeError("This platform does not provide 64-bit array doubles")
    reference_sum = math.fsum(values)
    shared = shared_memory.SharedMemory(create=True, size=len(values) * values.itemsize)
    try:
        with memoryview(values).cast("B") as source:
            shared.buf[:] = source
        del values
        print(f"Serial reference sum: {reference_sum:.17g}")
        print()
        print(f"{'Cores':>5} | {'Wall Time (ms)':>16} | {'Speedup':>10} | {'Amdahl Upper Bound':>18}", flush=True)
        print("-" * 60, flush=True)
        context = mp.get_context("spawn")
        baseline_ns = None
        for workers in worker_counts:
            total, elapsed_ns = benchmark(context, shared.name, ELEMENT_COUNT, workers)
            # Chunking changes floating-point rounding, so compare with tolerance.
            if not math.isclose(total, reference_sum, rel_tol=1e-14, abs_tol=1e-8):
                raise RuntimeError(f"Incorrect sum with {workers} workers: {total}")
            if baseline_ns is None:
                baseline_ns = elapsed_ns
            speedup = baseline_ns / elapsed_ns
            bound = 1 / (1 - PARALLEL_FRACTION + PARALLEL_FRACTION / workers)
            print(f"{workers:5d} | {elapsed_ns / 1_000_000:16.6f} | {speedup:9.4f}x | {bound:17.4f}x", flush=True)
        print("\nAll parallel sums matched the serial reference (rel_tol=1e-14, abs_tol=1e-8).")
    finally:
        shared.close()
        shared.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
