"""Compare static blocks and a dynamic work queue: python task4.py."""

import platform
import queue
import threading
import time

ITEM_COUNT = 1_000
WORKERS = 4
SLEEP_UNIT = 0.0001


def benchmark(dynamic):
    tasks = queue.Queue()
    if dynamic:
        for i in range(ITEM_COUNT):
            tasks.put(i)
    ready = threading.Barrier(WORKERS + 1)
    start_gate = threading.Event()
    completed = [[] for _ in range(WORKERS)]
    finish_ns = [0] * WORKERS

    def worker(worker_id):
        ready.wait()
        start_gate.wait()
        if dynamic:
            while True:
                try:
                    i = tasks.get_nowait()
                except queue.Empty:
                    break
                time.sleep(SLEEP_UNIT * (i % 20))
                completed[worker_id].append(i)
        else:
            begin = ITEM_COUNT * worker_id // WORKERS
            end = ITEM_COUNT * (worker_id + 1) // WORKERS
            for i in range(begin, end):
                time.sleep(SLEEP_UNIT * (i % 20))
                completed[worker_id].append(i)
        finish_ns[worker_id] = time.perf_counter_ns()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for thread in threads:
        thread.start()
    ready.wait()
    start_ns = time.perf_counter_ns()
    start_gate.set()
    for thread in threads:
        thread.join()
    end_ns = time.perf_counter_ns()

    # Check for skipped or duplicated work outside the timed interval.
    processed = sorted(item for items in completed for item in items)
    if processed != list(range(ITEM_COUNT)) or any(t == 0 for t in finish_ns):
        raise RuntimeError("Not all 1,000 items were processed exactly once")
    completion_ms = [(t - start_ns) / 1_000_000 for t in finish_ns]
    return {
        "wall_ms": (end_ns - start_ns) / 1_000_000,
        "first_ms": min(completion_ms),
        "last_ms": max(completion_ms),
        "delta_ms": (max(finish_ns) - min(finish_ns)) / 1_000_000,
        "completion_ms": completion_ms,
        "counts": [len(items) for items in completed],
        "requested_sleep_ms": [sum(i % 20 for i in items) * SLEEP_UNIT * 1000
                               for items in completed],
    }


def main():
    print("Student ID: 230103254")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"Items: {ITEM_COUNT} (indices 0..999); worker threads: {WORKERS}")
    print("Workload: time.sleep(0.0001 * (i % 20)) seconds for every item.")
    print("A: static contiguous blocks [0,250), [250,500), [500,750), [750,1000).")
    print("B: shared FIFO work queue, chunk size = 1.")
    print("Threads run sleeps concurrently; OS assigns them to cores (no affinity).")
    print("Wall time: start-gate release through all joins; excludes setup and validation.")
    print("Completion times: relative to the same start; delta = last minus first.")
    print("One measured run per strategy; actual sleep durations depend on OS scheduling.")
    print("The prescribed periodic costs give static blocks only a mild load imbalance.")
    results = [("A: Static", benchmark(False)), ("B: Dynamic", benchmark(True))]
    print()
    print(f"{'Strategy':<12} | {'Wall (ms)':>13} | {'First (ms)':>13} | {'Last (ms)':>13} | {'Delta (ms)':>13}")
    print("-" * 76)
    for label, result in results:
        print(f"{label:<12} | {result['wall_ms']:13.6f} | {result['first_ms']:13.6f} | "
              f"{result['last_ms']:13.6f} | {result['delta_ms']:13.6f}")
    for label, result in results:
        print(f"\n{label} per-worker details:")
        for i in range(WORKERS):
            print(f"  Worker {i}: items={result['counts'][i]}, "
                  f"requested sleep={result['requested_sleep_ms'][i]:.3f} ms, "
                  f"finished at={result['completion_ms'][i]:.6f} ms")
    print(f"\nWall-time ratio (Static / Dynamic): {results[0][1]['wall_ms'] / results[1][1]['wall_ms']:.4f}x")
    print("Validation passed: all 1,000 items processed exactly once in each strategy.")


if __name__ == "__main__":
    main()
