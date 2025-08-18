class NumberWithLogs {
    constructor(public result: number, public logs: string[] = []) {}

    static pure(x: number): NumberWithLogs {
        return new NumberWithLogs(x, []);
    }

    static of(result: number, ...logs: string[]): NumberWithLogs {
        return new NumberWithLogs(result, logs);
    }

    flatMap(transform: (x: number) => NumberWithLogs): NumberWithLogs {
        const next = transform(this.result);
        return new NumberWithLogs(next.result, this.logs.concat(next.logs));
    }
}

const square = (x: number): NumberWithLogs =>
    NumberWithLogs.of(x * x, `Squared ${x} to get ${x * x}.`);

const addOne = (x: number): NumberWithLogs =>
    NumberWithLogs.of(x + 1, `Added 1 to ${x} to get ${x + 1}.`);

const multiplyByThree = (x: number): NumberWithLogs =>
    NumberWithLogs.of(x * 3, `Multiplied ${x} by 3 to get ${x * 3}.`);

const result = NumberWithLogs.pure(5)
    .flatMap(addOne)
    .flatMap(square)
    .flatMap(multiplyByThree);

console.log(result);
