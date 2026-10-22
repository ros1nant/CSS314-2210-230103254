import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

/** Run: java task6.java (JDK 11 or later; standard library only). */
public class task6 {
    private static final int WORKERS = 4;
    private static final int ITERATIONS = 2_000_000;

    private static class Counter {
        // Deliberately neither volatile nor protected by a lock.
        int value = 0;
    }

    private static class Result {
        final int actual;
        final long elapsedNs;

        Result(int actual, long elapsedNs) {
            this.actual = actual;
            this.elapsedNs = elapsedNs;
        }
    }

    private static Result run(boolean synchronizedUpdates, int iterations) throws InterruptedException {
        Counter unsafe = new Counter();
        AtomicInteger atomic = new AtomicInteger(0);
        CountDownLatch ready = new CountDownLatch(WORKERS);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch finished = new CountDownLatch(WORKERS);
        Thread[] threads = new Thread[WORKERS];
        for (int i = 0; i < WORKERS; i++) {
            threads[i] = new Thread(() -> {
                ready.countDown();
                try {
                    start.await();
                    if (synchronizedUpdates) {
                        for (int j = 0; j < iterations; j++) {
                            atomic.incrementAndGet();
                        }
                    } else {
                        for (int j = 0; j < iterations; j++) {
                            unsafe.value++;
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    finished.countDown();
                }
            }, "incrementer-" + i);
            threads[i].start();
        }
        ready.await();
        long begin = System.nanoTime();
        start.countDown();
        finished.await();
        long elapsedNs = System.nanoTime() - begin;
        for (Thread thread : threads) thread.join();
        int actual = synchronizedUpdates ? atomic.get() : unsafe.value;
        if (synchronizedUpdates && actual != WORKERS * iterations) {
            throw new AssertionError("Atomic counter did not match the expected value");
        }
        return new Result(actual, elapsedNs);
    }

    private static void printResult(String name, Result result, int expected) {
        long lost = (long) expected - result.actual;
        double lostPercent = 100.0 * lost / expected;
        System.out.printf("%n%s%n", name);
        System.out.printf("Expected final value: %,d%n", expected);
        System.out.printf("Actual final value:   %,d%n", result.actual);
        System.out.printf("Updates lost:        %,d%n", lost);
        System.out.printf("Percentage lost:     %.6f%%%n", lostPercent);
        System.out.printf("Wall-clock time:     %.6f ms%n", result.elapsedNs / 1_000_000.0);
    }

    public static void main(String[] args) throws InterruptedException {
        Locale.setDefault(Locale.ROOT);
        System.out.println("Student ID: 230103254");
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("CPU / platform: " + System.getenv("PROCESSOR_IDENTIFIER"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Available logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.println("Worker threads: " + WORKERS);
        System.out.printf("Increment attempts per thread: %,d%n", ITERATIONS);
        System.out.println("Each run initializes its shared counter to zero.");
        System.out.println("Unsafe: plain int read-modify-write; no volatile, lock, or atomic increment.");
        System.out.println("Correct: AtomicInteger.incrementAndGet (atomic primitive, not a mutex).");
        System.out.println("Latches coordinate start/completion only; they do not protect unsafe updates.");
        System.out.println("Timing: start-gate release through completion; excludes creation, joins, and validation.");
        System.out.println("Warm-up: 3 pairs of 100,000 attempts per thread; one full measured run per version.");
        for (int i = 0; i < 3; i++) {
            run(false, 100_000);
            run(true, 100_000);
        }
        Result unsafe = run(false, ITERATIONS);
        Result atomic = run(true, ITERATIONS);
        int expected = WORKERS * ITERATIONS;
        printResult("UNSYNCHRONIZED", unsafe, expected);
        printResult("ATOMIC", atomic, expected);
        double penaltyMs = (atomic.elapsedNs - unsafe.elapsedNs) / 1_000_000.0;
        double ratio = (double) atomic.elapsedNs / unsafe.elapsedNs;
        System.out.printf("%nSynchronization time penalty (atomic - unsafe): %.6f ms%n", penaltyMs);
        System.out.printf("Atomic / unsafe wall-time ratio: %.4fx%n", ratio);
        System.out.printf("Wall-time increase: %.4f%%%n", (ratio - 1) * 100);
        System.out.println("Validation passed: atomic counter is exactly 8,000,000 (0% updates lost).");
        System.out.println("Unsafe results vary with scheduling and JIT optimization; lost updates are not guaranteed in every run.");
        System.out.println("The timing difference is an observed cost of correctness, not an isolated per-lock cost:");
        System.out.println("the unsafe data race can lose work and permits optimizations unavailable to the atomic version.");
    }
}
