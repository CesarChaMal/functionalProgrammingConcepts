import java.util.*;
import java.util.function.Function;

public class DemoMonadsImproved1 {
    record NumberWithLogs(int result, List<String> logs) {}

    static NumberWithLogs wrapWithLogs(int x) {
        return new NumberWithLogs(x, List.of());
    }

    static NumberWithLogs runWithLogs(NumberWithLogs input, Function<Integer, NumberWithLogs> transform) {
        var newNumberWithLogs = transform.apply(input.result);
        var combinedLogs = new ArrayList<>(input.logs);
        combinedLogs.addAll(newNumberWithLogs.logs);
        return new NumberWithLogs(newNumberWithLogs.result, combinedLogs);
    }

    static NumberWithLogs square(int x) {
        return new NumberWithLogs(x * x, List.of("Squared %d to get %d.".formatted(x, x * x)));
    }

    static NumberWithLogs addOne(int x) {
        return new NumberWithLogs(x + 1, List.of("Added 1 to %d to get %d.".formatted(x, x + 1)));
    }

    static NumberWithLogs multiplyByThree(int x) {
        return new NumberWithLogs(x * 3, List.of("Multiplied %d by 3 to get %d.".formatted(x, x * 3)));
    }

    public static void main(String[] args) {
        var a = wrapWithLogs(5);
        var b = runWithLogs(a, DemoMonadsImproved1::addOne);
        var c = runWithLogs(b, DemoMonadsImproved1::square);
        var d = runWithLogs(c, DemoMonadsImproved1::multiplyByThree);
        System.out.println(d);
    }
}