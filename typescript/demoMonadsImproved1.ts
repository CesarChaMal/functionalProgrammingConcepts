interface NumberWithLogs {
    result: number;
    logs: string[];
}

function wrapWithLogs(x: number): NumberWithLogs {
    return { result: x, logs: [] };
}

function runWithLogs(input: NumberWithLogs, transform: (x: number) => NumberWithLogs): NumberWithLogs {
    const newNumberWithLogs = transform(input.result);
    return {
        result: newNumberWithLogs.result,
        logs: input.logs.concat(newNumberWithLogs.logs)
    };
}

function square(x: number): NumberWithLogs {
    return {
        result: x * x,
        logs: [`Squared ${x} to get ${x * x}.`]
    };
}

function addOne(x: number): NumberWithLogs {
    return {
        result: x + 1,
        logs: [`Added 1 to ${x} to get ${x + 1}.`]
    };
}

function multiplyByThree(x: number): NumberWithLogs {
    return {
        result: x * 3,
        logs: [`Multiplied ${x} by 3 to get ${x * 3}.`]
    };
}

const a = wrapWithLogs(5);
const b = runWithLogs(a, addOne);
const c = runWithLogs(b, square);
const d = runWithLogs(c, multiplyByThree);
console.log(d);
