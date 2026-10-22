import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/** Run: java -Xmx2g task7.java (JDK 11+). Arrays occupy about 480 MB. */
public class task7 {
    private static final int SIZE = 30_000_000;
    private static final int SAMPLES = 3;
    private static volatile double sink;

    private static void kernel(double[] a, double[] b, int start, int end, boolean compute) {
        if (compute) {
            for (int i = start; i < end; i++) {
                b[i] = Math.sin(a[i]) * Math.cos(a[i]) + Math.tan(a[i]);
            }
        } else {
            for (int i = start; i < end; i++) b[i] = a[i];
        }
    }

    private static long timedRun(ThreadPoolExecutor pool, int workers,
                                 double[] a, double[] b, boolean compute) throws Exception {
        long start = System.nanoTime();
        Future<?>[] jobs = new Future<?>[workers];
        for (int worker = 0; worker < workers; worker++) {
            final int begin = (int) ((long) SIZE * worker / workers);
            final int end = (int) ((long) SIZE * (worker + 1) / workers);
            jobs[worker] = pool.submit(() -> kernel(a, b, begin, end, compute));
        }
        for (Future<?> job : jobs) job.get();
        return System.nanoTime() - start;
    }

    private static void validate(double[] a, double[] b, boolean compute) {
        // Validate all elements outside the measured interval.
        for (int i = 0; i < SIZE; i++) {
            double expected = compute
                    ? Math.sin(a[i]) * Math.cos(a[i]) + Math.tan(a[i]) : a[i];
            if (Math.abs(b[i] - expected) > 1e-12) {
                throw new AssertionError("Incorrect output at index " + i);
            }
        }
        sink = b[0] + b[SIZE - 1];
    }

    private static long measure(double[] a, double[] b, int workers, boolean compute) throws Exception {
        ThreadPoolExecutor pool = new ThreadPoolExecutor(workers, workers, 0L,
                TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>());
        pool.prestartAllCoreThreads();
        try {
            timedRun(pool, workers, a, b, compute); // Full-array warm-up, not reported.
            validate(a, b, compute);
            long[] times = new long[SAMPLES];
            for (int sample = 0; sample < SAMPLES; sample++) {
                times[sample] = timedRun(pool, workers, a, b, compute);
                validate(a, b, compute);
            }
            System.out.printf("Raw %s, workers=%d, ms: %.6f, %.6f, %.6f%n",
                    compute ? "B" : "A", workers,
                    times[0] / 1e6, times[1] / 1e6, times[2] / 1e6);
            Arrays.sort(times);
            return times[SAMPLES / 2];
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                pool.shutdownNow();
                throw new IllegalStateException("Worker pool failed to terminate");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int maxCores = Runtime.getRuntime().availableProcessors();
        int[] counts = Arrays.stream(new int[]{1, 2, maxCores}).distinct().toArray();
        System.out.println("Student ID: 230103254");
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("CPU / platform: " + System.getenv("PROCESSOR_IDENTIFIER"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Available logical cores: " + maxCores);
        System.out.println("Array size: 30,000,000 float64 elements; two arrays = 480,000,000 bytes.");
        System.out.println("Input: A[i] = ((i % 20001) - 10000) / 10000.0, in [-1, 1].");
        System.out.println("Kernel A (memory): B[i] = A[i].");
        System.out.println("Kernel B (compute): B[i] = sin(A[i]) * cos(A[i]) + tan(A[i]).");
        System.out.println("Threads: " + Arrays.toString(counts) + "; OS scheduling, no core affinity.");
        System.out.println("Timing includes dispatch and completion on a prestarted pool, excluding pool lifecycle,");
        System.out.println("allocation, initialization, one full-array warm-up per case, and validation.");
        System.out.println("Reported time is the median of 3 runs. Speedup = T(1)/T(p); efficiency = speedup/p.");
        System.out.println("Copy bandwidth is useful bytes (one 8-byte read + one 8-byte write) / time,");
        System.out.println("not measured RAM traffic; write allocation/cache effects may add traffic.");
        double[] a = new double[SIZE];
        double[] b = new double[SIZE];
        for (int i = 0; i < SIZE; i++) a[i] = ((i % 20001) - 10000) / 10000.0;
        long[][] times = new long[2][counts.length];
        for (int k = 0; k < 2; k++) {
            for (int c = 0; c < counts.length; c++) {
                times[k][c] = measure(a, b, counts[c], k == 1);
            }
        }
        System.out.printf("%n%-10s | %5s | %14s | %9s | %10s | %16s%n",
                "Kernel", "Cores", "Wall Time ms", "Speedup", "Efficiency", "Copy GB/s");
        System.out.println("-----------+-------+----------------+-----------+------------+-----------------");
        for (int k = 0; k < 2; k++) {
            for (int c = 0; c < counts.length; c++) {
                double speedup = (double) times[k][0] / times[k][c];
                String bandwidth = k == 0 ? String.format("%.4f", (double) SIZE * 16 / times[k][c]) : "n/a";
                System.out.printf("%-10s | %5d | %14.6f | %8.4fx | %10.4f | %16s%n",
                        k == 0 ? "A: Copy" : "B: Trig", counts[c], times[k][c] / 1e6,
                        speedup, speedup / counts[c], bandwidth);
            }
        }
        int last = counts.length - 1;
        double memorySpeedup = (double) times[0][0] / times[0][last];
        double computeSpeedup = (double) times[1][0] / times[1][last];
        System.out.println("\nScaling assessment:");
        System.out.println("Diagnostic rule: call copy scaling limited when max-core efficiency is below 50%.");
        if (memorySpeedup / counts[last] < 0.5) {
            System.out.println("Kernel A (vector copy) showed limited scaling, consistent with memory-channel bandwidth saturation.");
        } else {
            System.out.println("Kernel A did not meet the limited-scaling criterion; memory saturation was not demonstrated.");
        }
        System.out.printf("At %d workers: copy speedup=%.4fx; compute speedup=%.4fx.%n",
                counts[last], memorySpeedup, computeSpeedup);
        if (computeSpeedup > memorySpeedup) {
            System.out.println("Kernel B scaled better than copy, supporting a memory-bandwidth limit for Kernel A.");
        } else {
            System.out.println("Kernel B did not scale better than copy; this comparison does not isolate a memory-specific limit.");
        }
        System.out.println("Timing alone cannot prove RAM-channel saturation; hardware counters were not collected.");
        System.out.println("Scheduling, hybrid cores, SMT, and CPU frequency can also limit scaling.");
        System.out.println("Validation passed: all 30,000,000 outputs checked after every warm-up and measured run.");
    }
}
