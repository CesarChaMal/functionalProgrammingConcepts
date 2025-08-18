import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

public class DemoMonadsImproved2 {
    record NumberWithLogs(int result, List<String> logs) {
        static NumberWithLogs of(int result, String... logs) {
            return new NumberWithLogs(result, List.of(logs));
        }
        
        NumberWithLogs flatMap(Function<Integer, NumberWithLogs> transform) {
            var next = transform.apply(result);
            return new NumberWithLogs(next.result, 
                Stream.concat(logs.stream(), next.logs.stream()).toList());
        }
    }

    static NumberWithLogs pure(int x) {
        return new NumberWithLogs(x, List.of());
    }

    static NumberWithLogs square(int x) {
        return NumberWithLogs.of(x * x, "Squared %d to get %d.".formatted(x, x * x));
    }

    static NumberWithLogs addOne(int x) {
        return NumberWithLogs.of(x + 1, "Added 1 to %d to get %d.".formatted(x, x + 1));
    }

    static NumberWithLogs multiplyByThree(int x) {
        return NumberWithLogs.of(x * 3, "Multiplied %d by 3 to get %d.".formatted(x, x * 3));
    }

    public static void main(String[] args) {
        var result = pure(5)
            .flatMap(DemoMonadsImproved2::addOne)
            .flatMap(DemoMonadsImproved2::square)
            .flatMap(DemoMonadsImproved2::multiplyByThree);
        System.out.println(result);
    }
}