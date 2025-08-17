interface NumberWithLogs {
    result: number;
    logs: string[];
}

function square(x: number): NumberWithLogs {
    return {
        result: x * x,
        logs: [`Squared ${x} to get ${x * x}.`]
    };
}

function addOne(x: NumberWithLogs): NumberWithLogs {
    return {
        result: x.result + 1,
        logs: x.logs.concat([`Added 1 to ${x.result} to get ${x.result + 1}.`])
    };
}

const out = addOne(square(2));
console.log(out);
