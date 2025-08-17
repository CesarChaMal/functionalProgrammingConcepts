import java.util.*;

public class DemoMonadsModernJava {
    record NumberWithLogs(int result, List<String> logs) {}

    static NumberWithLogs square(int x) {
        return new NumberWithLogs(x * x, List.of("Squared %d to get %d.".formatted(x, x * x)));
    }

    static NumberWithLogs addOne(NumberWithLogs x) {
        var newLogs = new ArrayList<>(x.logs);
        newLogs.add("Added 1 to %d to get %d.".formatted(x.result, x.result + 1));
        return new NumberWithLogs(x.result + 1, newLogs);
    }

    public static void main(String[] args) {
        var out = addOne(square(2));
        System.out.println(out);
    }
}
