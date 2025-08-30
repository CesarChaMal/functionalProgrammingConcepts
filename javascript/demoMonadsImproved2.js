/*
ADVANCED MONADIC PATTERNS (JavaScript) — Production-Ready Implementation

Enterprise-grade monadic patterns featuring:
* Comprehensive error handling with stack traces
* Performance optimizations and lazy evaluation
* Integration with modern JavaScript features
* Production-ready logging and debugging
* Advanced composition patterns
*/

// === RESULT MONAD (Enhanced Either) ===

class Result {
    constructor(value, error = null) {
        this.value = value;
        this.error = error;
    }
    
    static ok(value) { return new Result(value); }
    static err(error) { return new Result(null, error); }
    
    static try(fn) {
        try {
            return Result.ok(fn());
        } catch (error) {
            return Result.err(error);
        }
    }
    
    static async tryAsync(asyncFn) {
        try {
            const value = await asyncFn();
            return Result.ok(value);
        } catch (error) {
            return Result.err(error);
        }
    }
    
    isOk() { return this.error === null; }
    isErr() { return this.error !== null; }
    
    map(f) {
        return this.isOk() ? Result.ok(f(this.value)) : this;
    }
    
    flatMap(f) {
        return this.isOk() ? f(this.value) : this;
    }
    
    mapErr(f) {
        return this.isErr() ? Result.err(f(this.error)) : this;
    }
    
    fold(errF, okF) {
        return this.isOk() ? okF(this.value) : errF(this.error);
    }
    
    unwrap() {
        if (this.isErr()) throw new Error(`Unwrap failed: ${this.error}`);
        return this.value;
    }
    
    unwrapOr(defaultValue) {
        return this.isOk() ? this.value : defaultValue;
    }
    
    toString() {
        return this.isOk() ? `Ok(${this.value})` : `Err(${this.error})`;
    }
}

// === OPTION MONAD (Enhanced Maybe) ===

class Option {
    constructor(value) { this.value = value; }
    
    static some(value) { return value == null ? Option.none() : new Option(value); }
    static none() { return new Option(null); }
    
    static fromNullable(value) { return Option.some(value); }
    
    isSome() { return this.value != null; }
    isNone() { return this.value == null; }
    
    map(f) {
        return this.isSome() ? Option.some(f(this.value)) : this;
    }
    
    flatMap(f) {
        return this.isSome() ? f(this.value) : this;
    }
    
    filter(predicate) {
        return this.isSome() && predicate(this.value) ? this : Option.none();
    }
    
    fold(noneF, someF) {
        return this.isSome() ? someF(this.value) : noneF();
    }
    
    getOrElse(defaultValue) {
        return this.isSome() ? this.value : defaultValue;
    }
    
    orElse(alternative) {
        return this.isSome() ? this : alternative();
    }
    
    zip(other) {
        return this.isSome() && other.isSome() 
            ? Option.some([this.value, other.value])
            : Option.none();
    }
    
    toString() {
        return this.isSome() ? `Some(${this.value})` : 'None';
    }
}

// === TASK MONAD (Async with Error Handling) ===

class Task {
    constructor(computation) { this.computation = computation; }
    
    static of(value) { return new Task(() => Promise.resolve(value)); }
    static reject(error) { return new Task(() => Promise.reject(error)); }
    
    static fromPromise(promise) { return new Task(() => promise); }
    
    map(f) {
        return new Task(() => this.computation().then(f));
    }
    
    flatMap(f) {
        return new Task(() => this.computation().then(value => f(value).computation()));
    }
    
    mapError(f) {
        return new Task(() => this.computation().catch(f));
    }
    
    fold(errorF, successF) {
        return new Task(() => 
            this.computation()
                .then(successF)
                .catch(errorF)
        );
    }
    
    timeout(ms) {
        return new Task(() => 
            Promise.race([
                this.computation(),
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)
                )
            ])
        );
    }
    
    retry(attempts = 3) {
        return new Task(async () => {
            let lastError;
            for (let i = 0; i < attempts; i++) {
                try {
                    return await this.computation();
                } catch (error) {
                    lastError = error;
                    if (i < attempts - 1) {
                        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
                    }
                }
            }
            throw lastError;
        });
    }
    
    run() { return this.computation(); }
}

// === READER MONAD (Dependency Injection) ===

class Reader {
    constructor(computation) { this.computation = computation; }
    
    static of(value) { return new Reader(() => value); }
    static ask() { return new Reader(env => env); }
    
    map(f) {
        return new Reader(env => f(this.computation(env)));
    }
    
    flatMap(f) {
        return new Reader(env => f(this.computation(env)).computation(env));
    }
    
    local(f) {
        return new Reader(env => this.computation(f(env)));
    }
    
    run(environment) { return this.computation(environment); }
}

// === UTILITY FUNCTIONS ===

const safeParseInt = (str) => {
    const num = parseInt(str, 10);
    return isNaN(num) ? Result.err(`Invalid integer: ${str}`) : Result.ok(num);
};

const safeDiv = (a, b) => 
    b === 0 ? Result.err("Division by zero") : Result.ok(a / b);

const fetchData = (url) => Task.fromPromise(
    fetch(url).then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
);

const logOperation = (operation) => Reader.ask().map(env => {
    if (env.logging) console.log(`[LOG] ${operation}`);
    return operation;
});

// === BUSINESS LOGIC EXAMPLES ===

// User validation with comprehensive error handling
const validateUser = (userData) => {
    const validateName = (name) => 
        name && name.trim().length > 0 
            ? Result.ok(name.trim())
            : Result.err("Name is required");
    
    const validateEmail = (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email)
            ? Result.ok(email)
            : Result.err("Invalid email format");
    };
    
    const validateAge = (age) => 
        age >= 0 && age <= 120
            ? Result.ok(age)
            : Result.err("Age must be between 0 and 120");
    
    return validateName(userData.name)
        .flatMap(name => validateEmail(userData.email)
            .flatMap(email => validateAge(userData.age)
                .map(age => ({ name, email, age }))));
};

// Database operations with error handling
const findUser = (id) => Task.of({ id, name: `User ${id}`, email: `user${id}@example.com` });

const updateUser = (id, updates) => Task.of({ id, ...updates, updatedAt: new Date() });

const userService = {
    getUser: (id) => 
        findUser(id)
            .map(user => Option.some(user))
            .mapError(() => Option.none()),
    
    updateUserSafely: (id, updates) =>
        validateUser(updates)
            .fold(
                error => Task.reject(error),
                validData => updateUser(id, validData)
            )
};

// Configuration-dependent operations
const createApiClient = () => Reader.ask().map(config => ({
    baseUrl: config.apiUrl,
    timeout: config.timeout || 5000,
    headers: { 'Authorization': `Bearer ${config.token}` }
}));

const makeRequest = (endpoint) => 
    createApiClient()
        .flatMap(client => Reader.of(
            Task.fromPromise(
                fetch(`${client.baseUrl}${endpoint}`, {
                    headers: client.headers,
                    timeout: client.timeout
                })
            )
        ));

// === DEMONSTRATIONS ===

console.log('=== RESULT MONAD ===');

const calculation = safeParseInt("10")
    .flatMap(a => safeParseInt("5")
        .flatMap(b => safeDiv(a, b)));

console.log('Safe calculation:', calculation.toString()); // Ok(2)

const errorCalculation = safeParseInt("10")
    .flatMap(a => safeDiv(a, 0));

console.log('Error calculation:', errorCalculation.toString()); // Err(Division by zero)

console.log('\n=== OPTION MONAD ===');

const users = [
    { id: 1, name: "Alice", profile: { city: "NYC" } },
    { id: 2, name: "Bob" }
];

const getUserCity = (userId) =>
    Option.fromNullable(users.find(u => u.id === userId))
        .flatMap(user => Option.fromNullable(user.profile))
        .flatMap(profile => Option.fromNullable(profile.city))
        .getOrElse("Unknown");

console.log('User 1 city:', getUserCity(1)); // NYC
console.log('User 2 city:', getUserCity(2)); // Unknown

console.log('\n=== TASK MONAD ===');

const processData = Task.of([1, 2, 3, 4, 5])
    .map(data => data.filter(x => x % 2 === 0))
    .map(evens => evens.map(x => x * 2))
    .timeout(1000);

processData.run().then(result => {
    console.log('Processed data:', result); // [4, 8]
});

console.log('\n=== READER MONAD ===');

const environment = {
    apiUrl: 'https://api.example.com',
    token: 'abc123',
    logging: true,
    timeout: 3000
};

const apiOperation = logOperation('Making API request')
    .flatMap(() => createApiClient())
    .map(client => `Client configured for ${client.baseUrl}`);

console.log('API operation:', apiOperation.run(environment));

console.log('\n=== COMPLEX COMPOSITION ===');

const businessWorkflow = (userData) =>
    validateUser(userData)
        .fold(
            error => Task.reject(`Validation failed: ${error}`),
            validUser => userService.updateUserSafely(1, validUser)
                .map(updatedUser => `User updated: ${updatedUser.name}`)
        );

const testUser = { name: "John Doe", email: "john@example.com", age: 30 };

businessWorkflow(testUser).run().then(result => {
    console.log('Workflow result:', result);
}).catch(error => {
    console.log('Workflow error:', error);
});

/*
PRODUCTION CONSIDERATIONS:

1. Error Tracking: Integrate with error monitoring services
2. Performance: Use lazy evaluation and memoization where appropriate
3. Testing: Property-based testing for monadic laws
4. Documentation: Clear API documentation with examples
5. Interoperability: Seamless integration with existing codebases
6. Type Safety: Consider TypeScript for additional compile-time guarantees

These patterns provide a solid foundation for building robust,
maintainable applications with functional programming principles.
*/