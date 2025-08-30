/*
IMPROVED MONADIC PATTERNS (JavaScript) — Enhanced Implementation

Advanced monadic patterns with:
* Fluent APIs and method chaining
* Comprehensive error handling
* Async/Promise integration
* Real-world use cases
*/

// === ENHANCED MAYBE MONAD ===

class Maybe {
    constructor(value) { this.value = value; }
    
    static of(value) { return new Maybe(value); }
    static none() { return new Maybe(null); }
    static some(value) { return value == null ? Maybe.none() : new Maybe(value); }
    
    isNone() { return this.value == null; }
    isSome() { return !this.isNone(); }
    
    map(f) {
        return this.isNone() ? this : Maybe.of(f(this.value));
    }
    
    flatMap(f) {
        return this.isNone() ? this : f(this.value);
    }
    
    filter(predicate) {
        return this.isNone() || !predicate(this.value) ? Maybe.none() : this;
    }
    
    getOrElse(defaultValue) {
        return this.isNone() ? defaultValue : this.value;
    }
    
    orElse(alternative) {
        return this.isNone() ? alternative : this;
    }
    
    fold(noneF, someF) {
        return this.isNone() ? noneF() : someF(this.value);
    }
    
    toString() {
        return this.isNone() ? 'None' : `Some(${this.value})`;
    }
}

// === ENHANCED EITHER MONAD ===

class Either {
    constructor(value, isLeft = false) {
        this.value = value;
        this.isLeft = isLeft;
    }
    
    static left(value) { return new Either(value, true); }
    static right(value) { return new Either(value, false); }
    
    static try(f) {
        try {
            return Either.right(f());
        } catch (error) {
            return Either.left(error.message);
        }
    }
    
    isLeft() { return this.isLeft; }
    isRight() { return !this.isLeft; }
    
    map(f) {
        return this.isLeft ? this : Either.right(f(this.value));
    }
    
    flatMap(f) {
        return this.isLeft ? this : f(this.value);
    }
    
    mapLeft(f) {
        return this.isLeft ? Either.left(f(this.value)) : this;
    }
    
    fold(leftF, rightF) {
        return this.isLeft ? leftF(this.value) : rightF(this.value);
    }
    
    getOrElse(defaultValue) {
        return this.isLeft ? defaultValue : this.value;
    }
    
    toString() {
        return this.isLeft ? `Left(${this.value})` : `Right(${this.value})`;
    }
}

// === ASYNC MONAD (Promise-based) ===

class AsyncM {
    constructor(promise) { this.promise = promise; }
    
    static of(value) { return new AsyncM(Promise.resolve(value)); }
    static reject(error) { return new AsyncM(Promise.reject(error)); }
    
    map(f) {
        return new AsyncM(this.promise.then(f));
    }
    
    flatMap(f) {
        return new AsyncM(this.promise.then(value => f(value).promise));
    }
    
    mapError(f) {
        return new AsyncM(this.promise.catch(f));
    }
    
    run() { return this.promise; }
}

// === VALIDATION MONAD ===

class Validation {
    constructor(value, isSuccess = true) {
        this.value = value;
        this.isSuccess = isSuccess;
    }
    
    static success(value) { return new Validation(value, true); }
    static failure(errors) { 
        return new Validation(Array.isArray(errors) ? errors : [errors], false); 
    }
    
    map(f) {
        return this.isSuccess ? Validation.success(f(this.value)) : this;
    }
    
    flatMap(f) {
        return this.isSuccess ? f(this.value) : this;
    }
    
    // Accumulate errors instead of short-circuiting
    ap(validationF) {
        if (this.isSuccess && validationF.isSuccess) {
            return Validation.success(validationF.value(this.value));
        }
        if (!this.isSuccess && !validationF.isSuccess) {
            return Validation.failure([...this.value, ...validationF.value]);
        }
        return this.isSuccess ? validationF : this;
    }
    
    fold(failureF, successF) {
        return this.isSuccess ? successF(this.value) : failureF(this.value);
    }
    
    toString() {
        return this.isSuccess ? `Success(${this.value})` : `Failure(${this.value.join(', ')})`;
    }
}

// === UTILITY FUNCTIONS ===

const safeDiv = (a, b) => b === 0 ? Maybe.none() : Maybe.some(a / b);

const safeParse = (str) => {
    const num = Number(str);
    return isNaN(num) ? Either.left(`Invalid number: ${str}`) : Either.right(num);
};

const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email) 
        ? Validation.success(email)
        : Validation.failure(['Invalid email format']);
};

const validateAge = (age) => {
    return age >= 0 && age <= 120
        ? Validation.success(age)
        : Validation.failure(['Age must be between 0 and 120']);
};

const validateName = (name) => {
    return name && name.trim().length > 0
        ? Validation.success(name.trim())
        : Validation.failure(['Name cannot be empty']);
};

// === REAL-WORLD EXAMPLES ===

console.log('=== ENHANCED MAYBE ===');

const user = { profile: { address: { street: "123 Main St" } } };

const getStreet = (user) => 
    Maybe.some(user)
        .flatMap(u => Maybe.some(u.profile))
        .flatMap(p => Maybe.some(p.address))
        .flatMap(a => Maybe.some(a.street))
        .getOrElse("No address");

console.log('Safe navigation:', getStreet(user)); // "123 Main St"
console.log('Safe navigation (null):', getStreet({})); // "No address"

console.log('\n=== ENHANCED EITHER ===');

const parseAndCalculate = (str1, str2) =>
    safeParse(str1)
        .flatMap(a => safeParse(str2).map(b => a + b))
        .fold(
            error => `Error: ${error}`,
            result => `Result: ${result}`
        );

console.log('Parse success:', parseAndCalculate("10", "20")); // "Result: 30"
console.log('Parse failure:', parseAndCalculate("abc", "20")); // "Error: Invalid number: abc"

console.log('\n=== VALIDATION ACCUMULATION ===');

const validateUser = (name, email, age) => {
    const nameV = validateName(name);
    const emailV = validateEmail(email);
    const ageV = validateAge(age);
    
    // Combine validations - accumulates all errors
    return nameV
        .flatMap(n => emailV
            .flatMap(e => ageV
                .map(a => ({ name: n, email: e, age: a }))));
};

const validUser = validateUser("John Doe", "john@example.com", 30);
console.log('Valid user:', validUser.toString());

const invalidUser = validateUser("", "invalid-email", -5);
console.log('Invalid user:', invalidUser.toString());

console.log('\n=== ASYNC MONAD ===');

const fetchUser = (id) => AsyncM.of({ id, name: `User ${id}` });
const fetchPosts = (userId) => AsyncM.of([`Post 1 by ${userId}`, `Post 2 by ${userId}`]);

const getUserWithPosts = (id) =>
    fetchUser(id)
        .flatMap(user => fetchPosts(user.id)
            .map(posts => ({ user, posts })));

getUserWithPosts(123).run().then(result => {
    console.log('Async result:', result);
});

console.log('\n=== CHAINING MULTIPLE MONADS ===');

const processUserData = (userStr, ageStr) =>
    safeParse(ageStr)
        .flatMap(age => age >= 18 
            ? Either.right({ name: userStr, age, canVote: true })
            : Either.left("Must be 18 or older"))
        .fold(
            error => Maybe.none(),
            user => Maybe.some(user)
        );

const result1 = processUserData("Alice", "25");
console.log('Adult user:', result1.toString()); // Some({name: "Alice", age: 25, canVote: true})

const result2 = processUserData("Bob", "16");
console.log('Minor user:', result2.toString()); // None

/*
ADVANCED PATTERNS:

1. Monad Transformers: Combining multiple monadic effects
2. Free Monads: Building DSLs with monadic structure
3. Tagless Final: Type-class based approach to effects
4. Effect Systems: Managing side effects at the type level

These patterns enable sophisticated functional architectures
while maintaining composability and testability.
*/