import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLongArray;

/** Run on Windows: java task3.java
 *  Or supply a verified physical core count: java task3.java 14
 *  Requires JDK 11 or later; no external dependencies.
 */
public class task3 {
    private static final long ITERATIONS = 50_000_000L;
    private static final int PADDED_STRIDE = 64 / Long.BYTES;

    private static int physicalCores() throws Exception {
        Process query = new ProcessBuilder("powershell.exe", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
                .redirectErrorStream(true).start();
        String output = new String(query.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
        if (query.waitFor() != 0) {
            throw new IllegalStateException("Physical-core query failed. Supply the verified count as an argument. " + output);
        }
        return Integer.parseInt(output);
    }

    private static long measure(int workers, int stride, long iterations) throws InterruptedException {
        // Unpadded: exactly one adjacent 8-byte counter per worker.
        // Padded: active counters are 64 bytes apart, regardless of array alignment.
        AtomicLongArray shared_counts = new AtomicLongArray(workers * stride);
        CountDownLatch ready = new CountDownLatch(workers);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch finished = new CountDownLatch(workers);
        Thread[] threads = new Thread[workers];
        for (int i = 0; i < workers; i++) {
            final int index = i * stride;
            threads[i] = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    // Identical atomic operations prevent JIT loop elimination in both cases.
                    for (long j = 0; j < iterations; j++) {
                        shared_counts.getAndIncrement(index);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    finished.countDown();
                }
            }, "counter-" + i);
            threads[i].start();
        }
        ready.await();
        long begin = System.nanoTime();
        start.countDown();
        finished.await();
        long elapsed = System.nanoTime() - begin;
        for (Thread thread : threads) {
            thread.join();
        }
        for (int i = 0; i < workers; i++) {
            if (shared_counts.get(i * stride) != iterations) {
                throw new AssertionError("Incorrect counter for worker " + i);
            }
        }
        return elapsed;
    }

    public static void main(String[] args) throws Exception {
        int workers = args.length == 0 ? physicalCores() : Integer.parseInt(args[0]);
        if (workers < 1) {
            throw new IllegalArgumentException("Physical core count must be positive");
        }
        System.out.println("Student ID: 230103254");
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical processors: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Physical cores / worker threads: " + workers);
        System.out.println("Core count source: " + (args.length == 0 ? "Windows CIM" : "command-line argument"));
        System.out.println("Increments per worker: " + ITERATIONS);
        System.out.println("Unpadded shared_counts length: " + workers);
        System.out.println("Padded active-counter spacing: 64 bytes (8 long elements)");
        System.out.println("Both versions use AtomicLongArray.getAndIncrement in every iteration.");
        System.out.println("Timing: start-gate release through completion; excludes creation and validation.");
        System.out.println("Warm-up: 100,000 increments per worker for each layout (not timed below).");
        System.out.println("One measured run per layout; OS schedules threads without core affinity.");
        measure(workers, 1, 100_000L);
        measure(workers, PADDED_STRIDE, 100_000L);
        System.out.println("Running Unpadded Shared...");
        long unpadded = measure(workers, 1, ITERATIONS);
        System.out.println("Running Padded...");
        long padded = measure(workers, PADDED_STRIDE, ITERATIONS);
        System.out.printf(java.util.Locale.ROOT, "%n%-20s | %18s%n", "Version", "Wall Time (ms)");
        System.out.println("---------------------+-------------------");
        System.out.printf(java.util.Locale.ROOT, "%-20s | %18.6f%n", "Unpadded Shared", unpadded / 1_000_000.0);
        System.out.printf(java.util.Locale.ROOT, "%-20s | %18.6f%n", "Padded (64 bytes)", padded / 1_000_000.0);
        System.out.printf(java.util.Locale.ROOT, "Execution ratio (Unpadded / Padded): %.4fx%n", (double) unpadded / padded);
        System.out.println("Validation passed: every counter equals 50,000,000 in both measured runs.");
        System.out.println("This measures the layout's timing effect, not hardware cache-invalidation events directly.");
    }
}
