import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

/** Run: java task5.java (JDK 11+, standard library only). */
public class task5 {
    private static final int[] SIZES = {100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000};
    private static final int SAMPLES = 7;
    private static final double A = 2.5;
    private static final double B = 1.25;
    private static volatile double sink;

    private static void transform(double[] x, double[] y, int begin, int end) {
        for (int i = begin; i < end; i++) {
            y[i] = A * x[i] + B;
        }
    }

    private static long serial(double[] x, double[] y) {
        long start = System.nanoTime();
        transform(x, y, 0, x.length);
        return System.nanoTime() - start;
    }

    private static long parallel(double[] x, double[] y, int workers) throws Exception {
        long start = System.nanoTime();
        ExecutorService pool = Executors.newFixedThreadPool(workers);
        try {
            Future<?>[] futures = new Future<?>[workers];
            for (int worker = 0; worker < workers; worker++) {
                final int begin = (int) ((long) x.length * worker / workers);
                final int end = (int) ((long) x.length * (worker + 1) / workers);
                futures[worker] = pool.submit(() -> transform(x, y, begin, end));
            }
            for (Future<?> future : futures) {
                future.get();
            }
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(1, TimeUnit.MINUTES)) {
                pool.shutdownNow();
                throw new IllegalStateException("Worker pool did not terminate");
            }
        }
        return System.nanoTime() - start;
    }

    private static void validate(double[] x, double[] serialY, double[] parallelY) {
        for (int i = 0; i < x.length; i++) {
            double expected = A * x[i] + B;
            if (serialY[i] != expected || parallelY[i] != expected) {
                throw new AssertionError("Incorrect output at index " + i);
            }
        }
        sink = serialY[x.length - 1] + parallelY[x.length - 1];
    }

    private static long median(long[] samples) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static String milliseconds(long[] samples) {
        StringBuilder text = new StringBuilder();
        for (long sample : samples) {
            if (text.length() > 0) text.append(", ");
            text.append(String.format(Locale.ROOT, "%.6f", sample / 1_000_000.0));
        }
        return text.toString();
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int workers = Runtime.getRuntime().availableProcessors();
        System.out.println("Student ID / input seed: 230103254");
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        System.out.println("CPU / platform: " + System.getenv("PROCESSOR_IDENTIFIER"));
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Available logical cores / pool workers: " + workers);
        System.out.println("Operation: y[i] = 2.5 * x[i] + 1.25; double (float64) arrays.");
        System.out.println("Parallel timing includes a fresh pool, thread startup, dispatch, computation, and shutdown.");
        System.out.println("Input/output allocation, input generation, warm-up, and validation are excluded.");
        System.out.println("Same transform loop in both modes; disjoint contiguous blocks for pool workers.");
        System.out.println("7 samples per mode; alternating execution order; comparison uses median wall time.");
        System.out.println("Warm-up: 12 serial/parallel pairs at N=1,000,000, then one pair at each tested N.");
        double[] warmX = new double[1_000_000];
        double[] warmS = new double[warmX.length];
        double[] warmP = new double[warmX.length];
        Arrays.fill(warmX, 0.75);
        for (int i = 0; i < 12; i++) {
            serial(warmX, warmS);
            parallel(warmX, warmP, workers);
            validate(warmX, warmS, warmP);
        }

        long[] serialMedians = new long[SIZES.length];
        long[] parallelMedians = new long[SIZES.length];
        long[][] serialSamples = new long[SIZES.length][SAMPLES];
        long[][] parallelSamples = new long[SIZES.length][SAMPLES];
        System.out.printf("%n%12s | %18s | %20s | %10s | %s%n",
                "N", "Serial median ms", "Parallel median ms", "Speedup", "Parallel vs serial");
        System.out.println("-------------+--------------------+----------------------+------------+-------------------");
        for (int k = 0; k < SIZES.length; k++) {
            int n = SIZES[k];
            double[] x = new double[n];
            double[] serialY = new double[n];
            double[] parallelY = new double[n];
            SplittableRandom random = new SplittableRandom(230103254L);
            for (int i = 0; i < n; i++) x[i] = random.nextDouble();
            serial(x, serialY);
            parallel(x, parallelY, workers);
            validate(x, serialY, parallelY);
            for (int sample = 0; sample < SAMPLES; sample++) {
                if (sample % 2 == 0) {
                    serialSamples[k][sample] = serial(x, serialY);
                    parallelSamples[k][sample] = parallel(x, parallelY, workers);
                } else {
                    parallelSamples[k][sample] = parallel(x, parallelY, workers);
                    serialSamples[k][sample] = serial(x, serialY);
                }
                validate(x, serialY, parallelY);
            }
            long s = serialMedians[k] = median(serialSamples[k]);
            long p = parallelMedians[k] = median(parallelSamples[k]);
            System.out.printf("%,12d | %18.6f | %20.6f | %9.4fx | %s%n",
                    n, s / 1_000_000.0, p / 1_000_000.0, (double) s / p,
                    p < s ? "FASTER" : p > s ? "SLOWER" : "EQUAL");
        }
        System.out.println("\nProgrammatic crossover detection (measured medians):");
        boolean found = false;
        for (int k = 1; k < SIZES.length; k++) {
            if (parallelMedians[k - 1] > serialMedians[k - 1]
                    && parallelMedians[k] < serialMedians[k]) {
                System.out.printf("SLOWER -> FASTER bracket: %,d < N <= %,d.%n", SIZES[k - 1], SIZES[k]);
                found = true;
            }
            if (parallelMedians[k - 1] < serialMedians[k - 1]
                    && parallelMedians[k] > serialMedians[k]) {
                System.out.printf("FASTER -> SLOWER reversal: between N=%,d and N=%,d.%n", SIZES[k - 1], SIZES[k]);
            }
        }
        if (!found) {
            System.out.println("No SLOWER -> FASTER crossover observed among the tested sizes.");
        }
        System.out.println("Brackets identify sampled endpoints, not an exact integer threshold or every interior N.");
        System.out.println("Results describe this fresh-pool benchmark; scheduling, JIT, and memory effects can vary between runs.");
        System.out.println("\nRaw wall-time samples (ms), in sample order:");
        for (int k = 0; k < SIZES.length; k++) {
            System.out.println("N=" + SIZES[k] + " serial:   " + milliseconds(serialSamples[k]));
            System.out.println("N=" + SIZES[k] + " parallel: " + milliseconds(parallelSamples[k]));
        }
        System.out.println("Validation passed: every output element matched a*x[i]+b in all runs.");
    }
}
