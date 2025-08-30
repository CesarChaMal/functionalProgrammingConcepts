/*
MONADIC PATTERNS (JavaScript) — Basic Implementation

Demonstrates core monadic patterns:
* Maybe monad for null safety
* Either monad for error handling  
* IO monad for side effects
* State monad for stateful computation
*/

// === MAYBE MONAD ===

class Maybe {
    constructor(value) { this.value = value; }
    
    static of(value) { return new Maybe(value); }
    static none() { return new Maybe(null); }
    static some(value) { return new Maybe(value); }
    
    isNone() { return this.value === null || this.value === undefined; }
    
    map(f) {
        return this.isNone() ? Maybe.none() : Maybe.of(f(this.value));
    }
    
    flatMap(f) {
        return this.isNone() ? Maybe.none() : f(this.value);
    }
    
    getOrElse(defaultValue) {
        return this.isNone() ? defaultValue : this.value;
    }
}

// === EITHER MONAD ===

class Either {
    constructor(value, isLeft = false) {
        this.value = value;
        this.isLeft = isLeft;
    }
    
    static left(value) { return new Either(value, true); }
    static right(value) { return new Either(value, false); }
    
    map(f) {
        return this.isLeft ? this : Either.right(f(this.value));
    }
    
    flatMap(f) {
        return this.isLeft ? this : f(this.value);
    }
    
    fold(leftF, rightF) {
        return this.isLeft ? leftF(this.value) : rightF(this.value);
    }
}

// === IO MONAD ===

class IO {
    constructor(effect) { this.effect = effect; }
    
    static of(value) { return new IO(() => value); }
    
    map(f) {
        return new IO(() => f(this.effect()));
    }
    
    flatMap(f) {
        return new IO(() => f(this.effect()).effect());
    }
    
    run() { return this.effect(); }
}

// === STATE MONAD ===

class State {
    constructor(runState) { this.runState = runState; }
    
    static of(value) {
        return new State(state => [value, state]);
    }
    
    map(f) {
        return new State(state => {
            const [value, newState] = this.runState(state);
            return [f(value), newState];
        });
    }
    
    flatMap(f) {
        return new State(state => {
            const [value, newState] = this.runState(state);
            return f(value).runState(newState);
        });
    }
    
    run(initialState) {
        return this.runState(initialState);
    }
}

// === UTILITY FUNCTIONS ===

const safeDiv = (a, b) => b === 0 ? Maybe.none() : Maybe.some(a / b);

const parseInt = (str) => {
    const num = Number(str);
    return isNaN(num) ? Either.left(`Invalid number: ${str}`) : Either.right(num);
};

const readLine = (prompt) => new IO(() => {
    // Simulate reading input
    console.log(prompt);
    return "42"; // Mock input
});

const println = (msg) => new IO(() => {
    console.log(msg);
    return msg;
});

const get = () => new State(state => [state, state]);
const put = (newState) => new State(() => [null, newState]);

// === DEMONSTRATIONS ===

console.log('=== MAYBE MONAD ===');

const maybeResult = Maybe.of(10)
    .flatMap(x => safeDiv(x, 2))
    .flatMap(x => safeDiv(x, 0))
    .map(x => x * 2)
    .getOrElse("Division by zero");

console.log('Safe division chain:', maybeResult); // "Division by zero"

const maybeSuccess = Maybe.of(20)
    .flatMap(x => safeDiv(x, 4))
    .map(x => x + 1)
    .getOrElse("Error");

console.log('Successful chain:', maybeSuccess); // 6

console.log('\n=== EITHER MONAD ===');

const eitherResult = parseInt("10")
    .flatMap(x => parseInt("5"))
    .map(y => x => x + y)
    .fold(
        error => `Error: ${error}`,
        value => `Success: ${value}`
    );

console.log('Either parsing:', eitherResult); // Success: function

const eitherError = parseInt("abc")
    .flatMap(x => parseInt("5"))
    .fold(
        error => `Error: ${error}`,
        value => `Success: ${value}`
    );

console.log('Either error:', eitherError); // Error: Invalid number: abc

console.log('\n=== IO MONAD ===');

const ioProgram = readLine("Enter a number:")
    .flatMap(input => parseInt(input).fold(
        error => IO.of(`Error: ${error}`),
        num => IO.of(num * 2)
    ))
    .flatMap(result => println(`Result: ${result}`));

console.log('IO program result:', ioProgram.run());

console.log('\n=== STATE MONAD ===');

const counter = get()
    .flatMap(count => put(count + 1))
    .flatMap(() => get())
    .map(count => count * 2);

const [result, finalState] = counter.run(5);
console.log('State computation:', { result, finalState }); // { result: 12, finalState: 6 }

/*
MONADIC LAWS VERIFICATION:

1. Left Identity: M.of(a).flatMap(f) === f(a)
2. Right Identity: m.flatMap(M.of) === m  
3. Associativity: m.flatMap(f).flatMap(g) === m.flatMap(x => f(x).flatMap(g))

These laws ensure monads compose predictably and maintain referential transparency.
*/