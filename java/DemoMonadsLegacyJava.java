import java.util.*;

public class DemoMonadsLegacyJava {
    static class NumberWithLogs {
        int result;
        List<String> logs;
        NumberWithLogs(int result, List<String> logs) {
            this.result = result;
            this.logs = logs;
        }
    }

    static NumberWithLogs square(int x) {
        return new NumberWithLogs(x * x,
            List.of("Squared " + x + " to get " + (x * x) + "."));
    }

    static NumberWithLogs addOne(NumberWithLogs x) {
        List<String> newLogs = new ArrayList<>(x.logs);
        newLogs.add("Added 1 to " + x.result + " to get " + (x.result + 1) + ".");
        return new NumberWithLogs(x.result + 1, newLogs);
    }

    public static void main(String[] args) {
        NumberWithLogs out = addOne(square(2));
        System.out.println("Result: " + out.result);
        System.out.println("Logs: " + out.logs);
    }
}
