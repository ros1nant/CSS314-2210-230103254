"""CPU IPC transfer benchmark: python task4p.py (requires numpy).

Path A pickles an ndarray through a process pipe. Path B sends only a shared
memory descriptor; the receiver attaches a zero-copy ndarray view.
"""

import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import platform
import statistics
import time

import numpy as np

STUDENT_ID = 230103254
SIZES = (100_000, 1_000_000, 10_000_000)
REPEATS = 5


def receiver(connection):
    connection.send(("started",))
    try:
        while True:
            message = connection.recv()
            if message[0] == "stop":
                return
            shared = None
            array = None
            try:
                if message[0] == "pickle":
                    # recv() has already unpickled and materialized the array.
                    array = message[1]
                elif message[0] == "shared":
                    _, name, shape, dtype = message
                    shared = shared_memory.SharedMemory(name=name)
                    array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shared.buf)
                else:
                    raise ValueError("Unknown transfer mode")
                array.setflags(write=False)
                connection.send(("ready", array.size, array.dtype.str,
                                 float(array[0]), float(array[-1])))
                # Validation is a separate message, excluded from the timed ACK.
                checksum = float(np.sum(array, dtype=np.float64))
                array = None
                message = None
                if shared is not None:
                    shared.close()
                    shared = None
                connection.send(("validated", checksum))
            except Exception as error:
                connection.send(("error", repr(error)))
                raise
            finally:
                array = None
                if shared is not None:
                    shared.close()
    finally:
        connection.close()


def receive(connection):
    if not connection.poll(60):
        raise TimeoutError("Receiver did not respond within 60 seconds")
    response = connection.recv()
    if response[0] == "error":
        raise RuntimeError(response[1])
    return response


def transfer(connection, message, array, checksum):
    start = time.perf_counter_ns()
    connection.send(message)
    acknowledgement = receive(connection)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    expected = ("ready", array.size, array.dtype.str, float(array[0]), float(array[-1]))
    if acknowledgement != expected:
        raise AssertionError(f"Invalid receiver acknowledgement: {acknowledgement}")
    validation = receive(connection)
    if validation[0] != "validated" or not math.isclose(validation[1], checksum, rel_tol=1e-14):
        raise AssertionError("Receiver checksum differs from the source array")
    return elapsed_ms


def main():
    print(f"Student ID / seed: {STUDENT_ID}")
    print(f"OS: {platform.platform()}")
    print(f"CPU / platform: {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Python: {platform.python_version()}; NumPy: {np.__version__}")
    print("CPU-only benchmark: parent process sends float64 arrays to one spawned receiver.")
    print("Path A: Connection.send(ndarray) -> pickle + pipe transfer + receiver unpickle.")
    print("Path B: send name/shape/dtype -> shared-memory attach + zero-copy ndarray view.")
    print("Timed interval: sender send() through receiver-ready ACK, including metadata and endpoint checks.")
    print("Full-array receiver checksum and detach complete outside the measured interval, before the next trial.")
    print("Process startup, random input generation, and warm-up are excluded.")
    print("Shared setup = allocation/mapping + initial payload copy; shown separately, not called zero-copy.")
    print("B setup + access estimates a first-use cost; repeated access amortizes the initial copy.")
    print("One warm-up per path/size; 5 measured trials per path, alternating order; table uses medians.")

    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=receiver, args=(child,))
    process.start()
    child.close()
    results = []
    try:
        if receive(parent) != ("started",):
            raise RuntimeError("Receiver startup failed")
        rng = np.random.default_rng(STUDENT_ID)
        for size in SIZES:
            array = rng.random(size, dtype=np.float64)
            checksum = float(np.sum(array, dtype=np.float64))
            setup_start = time.perf_counter_ns()
            shared = shared_memory.SharedMemory(create=True, size=array.nbytes)
            view = None
            try:
                view = np.ndarray(array.shape, dtype=array.dtype, buffer=shared.buf)
                np.copyto(view, array)
                setup_ms = (time.perf_counter_ns() - setup_start) / 1_000_000
                view = None
                messages = {
                    "A": ("pickle", array),
                    "B": ("shared", shared.name, array.shape, array.dtype.str),
                }
                for path in ("A", "B"):
                    transfer(parent, messages[path], array, checksum)
                samples = {"A": [], "B": []}
                for trial in range(REPEATS):
                    order = ("A", "B") if trial % 2 == 0 else ("B", "A")
                    for path in order:
                        samples[path].append(transfer(parent, messages[path], array, checksum))
                a_ms = statistics.median(samples["A"])
                b_ms = statistics.median(samples["B"])
                results.append((size, array.nbytes, a_ms, b_ms, setup_ms))
                print(f"\nN={size:,}, receiver checksum={checksum:.17g}")
                for path in ("A", "B"):
                    print(f"Path {path} raw times (ms): " + ", ".join(f"{t:.6f}" for t in samples[path]))
            finally:
                view = None
                shared.close()
                shared.unlink()
        parent.send(("stop",))
        process.join(30)
        if process.exitcode != 0:
            raise RuntimeError(f"Receiver exit code: {process.exitcode}")
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join()
        process.close()

    print("\nTransfer overhead (median wall-clock milliseconds):")
    print(f"{'N':>12} | {'Bytes':>12} | {'A: Pickle IPC':>14} | {'B: Shared access':>16} | {'Shared setup':>13} | {'B setup+access':>15} | {'A/B':>9}")
    for size, nbytes, a_ms, b_ms, setup_ms in results:
        print(f"{size:12,d} | {nbytes:12,d} | {a_ms:14.6f} | {b_ms:16.6f} | "
              f"{setup_ms:13.6f} | {setup_ms + b_ms:15.6f} | {a_ms / b_ms:8.2f}x")
    print("Validation PASS: receiver shape, dtype, endpoints, and full-array checksum matched for every trial.")
    print("Shared access avoids payload serialization/copying, but reading the payload later still incurs memory traffic.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
