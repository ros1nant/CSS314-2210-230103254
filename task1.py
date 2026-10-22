"""Serial baseline. Run with: python task1.py (standard library only)."""

from array import array
import math
import os
import platform
import random
import time


STUDENT_ID = "230103254"
ELEMENT_COUNT = 20_000_000


def serial_statistics(values):
    """Return the sum and population standard deviation using serial passes."""
    total = math.fsum(values)
    mean = total / len(values)
    squared_deviations = math.fsum((value - mean) ** 2 for value in values)
    return total, math.sqrt(squared_deviations / len(values))


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / platform details: {platform.processor() or platform.machine()}")
    print(f"Total logical cores (os.cpu_count): {os.cpu_count()}")
    print("Execution: serial, single thread")
    print(f"Seed: {int(STUDENT_ID)}")
    print(f"Array: {ELEMENT_COUNT:,} float64 elements, uniform [0, 1)")

    start = time.perf_counter()
    rng = random.Random(int(STUDENT_ID))
    values = array("d", (rng.random() for _ in range(ELEMENT_COUNT)))
    if values.itemsize != 8:
        raise RuntimeError("This platform does not provide 64-bit array doubles")
    generated = time.perf_counter()

    # No worker threads, processes, or threaded numerical libraries are used.
    total, standard_deviation = serial_statistics(values)
    finished = time.perf_counter()

    print(f"Sum: {total:.17g}")
    print(f"Population standard deviation (ddof=0): {standard_deviation:.17g}")
    print(f"Array generation wall-clock time (ms): {(generated - start) * 1000:.3f}")
    print(f"Serial computation wall-clock time (ms): {(finished - generated) * 1000:.3f}")
    print(f"Total wall-clock time (ms): {(finished - start) * 1000:.3f}")


if __name__ == "__main__":
    main()
