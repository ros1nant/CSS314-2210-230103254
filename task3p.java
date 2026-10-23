import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Run: java task3p.java (JDK 11+, standard library only). */
public class task3p {
    private static final long SEED = 230103254L;
    private static final int SIZE = 2_000_000;
    private static final int REPEATS = 7;

    private static long serialScan(int[] input, int[] output) {
        long start = System.nanoTime();
        int sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += input[i];
            output[i] = sum;
        }
        return System.nanoTime() - start;
    }

    private static long parallelScan(int[] input, int[] output, int workers,
                                      ThreadPoolExecutor pool) throws Exception {
        long start = System.nanoTime();
        int[] chunkSums = new int[workers];
        int[] offsets = new int[workers];
        Future<?>[] jobs = new Future<?>[workers];
        // Pass 1: inclusive prefix sums within disjoint contiguous chunks.
        for (int worker = 0; worker < workers; worker++) {
            final int index = worker;
            final int begin = (int) ((long) input.length * worker / workers);
            final int end = (int) ((long) input.length * (worker + 1) / workers);
            jobs[worker] = pool.submit(() -> {
                int sum = 0;
                for (int i = begin; i < end; i++) {
                    sum += input[i];
                    output[i] = sum;
                }
                chunkSums[index] = sum;
            });
        }
        for (Future<?> job : jobs) job.get();
        // Exclusive scan of chunk totals supplies the preceding chunks' sum.
        for (int worker = 1; worker < workers; worker++) {
            offsets[worker] = offsets[worker - 1] + chunkSums[worker - 1];
        }
        // Pass 2: add each chunk's offset to every local prefix value.
        for (int worker = 0; worker < workers; worker++) {
            final int offset = offsets[worker];
            final int begin = (int) ((long) input.length * worker / workers);
            final int end = (int) ((long) input.length * (worker + 1) / workers);
            jobs[worker] = pool.submit(() -> {
                for (int i = begin; i < end; i++) output[i] += offset;
            });
        }
        for (Future<?> job : jobs) job.get();
        return System.nanoTime() - start;
    }

    private static void verify(int[] serial, int[] parallel) {
        // Explicit assertion remains active without Java's optional -ea flag.
        if (parallel[parallel.length - 1] != serial[serial.length - 1]) {
            throw new AssertionError("parallel_output[-1] != serial_output[-1]");
        }
        if (!Arrays.equals(serial, parallel)) {
            throw new AssertionError("Intermediate prefix elements differ");
        }
    }

    private static long median(long[] times) {
        long[] sorted = times.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static void printSamples(String name, long[] times) {
        System.out.print(name + " raw times (ms): ");
        for (int i = 0; i < times.length; i++) {
            System.out.printf("%s%.6f", i == 0 ? "" : ", ", times[i] / 1e6);
        }
        System.out.println();
    }

    private static void benchmark() throws Exception {
        int workers = Runtime.getRuntime().availableProcessors();
        System.out.println("Student ID / seed: " + SEED);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical cores / worker threads: " + workers);
        System.out.println("Input: 2,000,000 integers, SplittableRandom(seed).nextInt(1, 11).");
        System.out.println("Maximum possible prefix = 20,000,000; signed 32-bit integers suffice.");
        System.out.println("Parallel algorithm: local inclusive scans, serial chunk-offset reduction, parallel offset addition.");
        System.out.println("Work: O(N + p); ideal parallel span: O(N/p + p), excluding scheduling costs.");
        System.out.println("Timing includes both passes, offset reduction, task dispatch, and completion.");
        System.out.println("Input/output allocation, generation, pool startup/shutdown, warm-up, and checks are excluded.");
        System.out.println("5 full warm-up pairs; 7 measured pairs with alternating order; table uses medians.");
        int[] input = new int[SIZE];
        int[] serial = new int[SIZE];
        int[] parallel = new int[SIZE];
        SplittableRandom random = new SplittableRandom(SEED);
        long expectedTotal = 0;
        for (int i = 0; i < SIZE; i++) {
            input[i] = random.nextInt(1, 11);
            expectedTotal += input[i];
        }
        ThreadPoolExecutor pool = new ThreadPoolExecutor(workers, workers, 0L,
                TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>());
        pool.prestartAllCoreThreads();
        try {
            // Check uneven and empty chunk boundaries as well as the large input.
            for (int[] small : new int[][]{{1, 2, 3, 4, 5, 6, 7}, {10}}) {
                int[] s = new int[small.length];
                int[] p = new int[small.length];
                serialScan(small, s);
                parallelScan(small, p, 4, pool);
                verify(s, p);
            }
            for (int i = 0; i < 5; i++) {
                serialScan(input, serial);
                parallelScan(input, parallel, workers, pool);
                verify(serial, parallel);
            }
            long[] serialTimes = new long[REPEATS];
            long[] parallelTimes = new long[REPEATS];
            for (int i = 0; i < REPEATS; i++) {
                if (i % 2 == 0) {
                    serialTimes[i] = serialScan(input, serial);
                    parallelTimes[i] = parallelScan(input, parallel, workers, pool);
                } else {
                    parallelTimes[i] = parallelScan(input, parallel, workers, pool);
                    serialTimes[i] = serialScan(input, serial);
                }
                verify(serial, parallel);
                if (serial[SIZE - 1] != expectedTotal) {
                    throw new AssertionError("Final prefix differs from input total");
                }
            }
            printSamples("Serial", serialTimes);
            printSamples("Parallel", parallelTimes);
            long s = median(serialTimes);
            long p = median(parallelTimes);
            System.out.println("\nSerial Time (ms) | Parallel Time (ms) | Speedup | Verification Status");
            System.out.printf("%.6f | %.6f | %.4fx | PASS%n", s / 1e6, p / 1e6, (double) s / p);
            System.out.println("serial_output[-1]: " + serial[SIZE - 1]);
            System.out.println("parallel_output[-1]: " + parallel[SIZE - 1]);
            System.out.println("PASS: final-element assertion and all 2,000,000 prefix values match.");
            System.out.println("PASS: uneven/empty chunk boundary checks.");
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                pool.shutdownNow();
                throw new IllegalStateException("Pool failed to terminate");
            }
        }
    }

    public static void main(String[] args) {
        Locale.setDefault(Locale.ROOT);
        try {
            benchmark();
        } catch (Exception | AssertionError failure) {
            System.out.println("Verification Status: FAIL");
            System.err.println(failure);
            System.exit(1);
        }
    }
}
