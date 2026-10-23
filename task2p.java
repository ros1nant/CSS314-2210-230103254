import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;

/** Run: java task2p.java (JDK 11+, standard library only). */
public class task2p {
    private static final long STUDENT_ID = 230103254L;
    private static final int THREADS = 4;
    private static final long POINTS = 40_000_000L;

    private static class Result {
        final long inside;
        final long elapsedNs;

        Result(long inside, long elapsedNs) {
            this.inside = inside;
            this.elapsedNs = elapsedNs;
        }
    }

    private static Result run(boolean sharedMode, long points) throws InterruptedException {
        // Both modes use the same PRNG algorithm. SplittableRandom itself has no
        // internal locking; every access to the shared instance is guarded below.
        SplittableRandom shared = sharedMode ? new SplittableRandom(STUDENT_ID) : null;
        Object generatorLock = new Object();
        long[] hits = new long[THREADS];
        long[] processed = new long[THREADS];
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch start = new CountDownLatch(1);
        Thread[] workers = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            final int index = i;
            final long count = points * (i + 1) / THREADS - points * i / THREADS;
            workers[i] = new Thread(() -> {
                SplittableRandom local = sharedMode ? null
                        : new SplittableRandom(STUDENT_ID * 10 + index);
                ready.countDown();
                try {
                    start.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                long inside = 0;
                if (sharedMode) {
                    for (long point = 0; point < count; point++) {
                        double x;
                        double y;
                        synchronized (generatorLock) {
                            // One lock acquisition protects both coordinates of a point.
                            x = shared.nextDouble();
                            y = shared.nextDouble();
                        }
                        if (x * x + y * y <= 1.0) inside++;
                    }
                } else {
                    for (long point = 0; point < count; point++) {
                        double x = local.nextDouble();
                        double y = local.nextDouble();
                        if (x * x + y * y <= 1.0) inside++;
                    }
                }
                // Only one final publication per worker, avoiding shared hit-counter contention.
                hits[index] = inside;
                processed[index] = count;
            }, "monte-carlo-" + i);
            workers[i].start();
        }
        ready.await();
        long begin = System.nanoTime();
        start.countDown();
        for (Thread worker : workers) worker.join();
        long inside = 0;
        for (long hit : hits) inside += hit;
        long elapsedNs = System.nanoTime() - begin;
        long totalProcessed = 0;
        for (long count : processed) totalProcessed += count;
        if (totalProcessed != points || inside < 0 || inside > points) {
            throw new AssertionError("Incomplete or invalid Monte Carlo run");
        }
        return new Result(inside, elapsedNs);
    }

    private static void printResult(String name, Result result) {
        double pi = 4.0 * result.inside / POINTS;
        System.out.printf("%n%s%n", name);
        System.out.printf("Sample points: %,d%n", POINTS);
        System.out.printf("Points inside quarter circle: %,d%n", result.inside);
        System.out.printf("Computed pi: %.12f%n", pi);
        System.out.printf("Absolute error from Math.PI: %.12f%n", Math.abs(pi - Math.PI));
        System.out.printf("Wall time (ms): %.6f%n", result.elapsedNs / 1_000_000.0);
    }

    public static void main(String[] args) throws InterruptedException {
        Locale.setDefault(Locale.ROOT);
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("CPU / platform: " + System.getenv("PROCESSOR_IDENTIFIER"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads per mode: " + THREADS);
        System.out.printf("Samples per mode: %,d (%,d per thread)%n", POINTS, POINTS / THREADS);
        System.out.println("PRNG algorithm for both modes: java.util.SplittableRandom.");
        System.out.println("Shared seed: " + STUDENT_ID + "; synchronized mutex around each x,y pair.");
        for (int i = 0; i < THREADS; i++) {
            System.out.println("Independent seed for thread " + i + ": " + (STUDENT_ID * 10 + i));
        }
        System.out.println("Samples: x,y uniform in [0,1); pi = 4 * count(x*x+y*y <= 1) / samples.");
        System.out.println("Timing includes start release, sampling, joins, and hit reduction;");
        System.out.println("excludes thread/PRNG creation, warm-up, validation, and printing.");
        System.out.println("Warm-up: 1,000,000 points per mode; measured runs restart with fresh PRNG seeds.");
        System.out.println("One measured run per mode; each completes all 40,000,000 points.");
        run(true, 1_000_000);
        run(false, 1_000_000);
        System.out.println("Running shared PRNG...");
        Result shared = run(true, POINTS);
        printResult("Shared PRNG", shared);
        System.out.println("\nRunning independent PRNGs...");
        Result independent = run(false, POINTS);
        printResult("Independent PRNG", independent);
        System.out.printf("%nSpeedup from PRNG state isolation (Shared / Independent): %.4fx%n",
                (double) shared.elapsedNs / independent.elapsedNs);
        System.out.println("Validation passed: each mode processed exactly 40,000,000 points.");
        System.out.println("Streams differ by design, so pi estimates need not match; timing ratios vary by run.");
    }
}
