# CLAUDE.md - AI Assistant Guide

## Project Overview

This is a **multi-language functional programming concepts repository** that provides comprehensive, well-documented examples of 16 fundamental FP concepts across **Scala**, **Java**, **Python**, **TypeScript**, and **JavaScript**.

### Purpose
Educational reference implementation demonstrating:
- Core FP concepts (lambdas, composition, functors, monads, etc.)
- Advanced FP patterns (natural transformations, monoids, ADTs, lazy evaluation)
- Cross-language comparison of FP idioms
- Mathematical laws and property-based testing
- Production-ready patterns and optimizations

---

## Repository Structure

```
functionalProgrammingConcepts/
├── scala/                    # Scala implementations (primary language)
│   ├── FPAnnotated.scala     # Complete 16-concept guide with Cats library
│   ├── DemoMonads*.scala     # Monadic examples with fluent APIs
│   └── LambdaCalculus.scala  # Lambda calculus demonstrations
├── java/                     # Java 17+ implementations
│   ├── FPJavaAnnotated.java  # Complete guide using records, sealed classes
│   ├── DemoMonads*.java      # Monadic patterns
│   └── LambdaCalculus.java
├── python/                   # Python 3.10+ implementations
│   ├── fp_annotated.py       # Guide using dataclasses, protocols, match
│   ├── demo_monads*.py       # Monadic patterns
│   └── lambda_calculus.py
├── typescript/               # TypeScript implementations
│   ├── fpAnnotated.ts        # Guide with advanced types, conditional types
│   ├── demoMonads*.ts        # Monadic patterns with classes
│   └── lambdaCalculus.ts
├── javascript/               # JavaScript implementations
│   ├── fpAnnotated.js        # Modern JavaScript examples
│   ├── demoMonads*.js
│   └── lambdaCalculus.js
├── build.sbt                 # SBT build configuration (Scala 2.13.14)
├── pom.xml                   # Maven configuration (Java 21, mixed Scala/Java)
├── setup_scala.sh            # Scala environment setup script
├── .gitignore                # Comprehensive ignore rules for all languages
└── README.md                 # User-facing documentation
```

---

## The 16 Core FP Concepts

### Core Concepts (1-8)
1. **Lambda, Application, Currying, Partial Application** — Function fundamentals
2. **Composition (∘)** — Combining functions
3. **Referential Transparency** — Pure expressions
4. **Immutability** — Unchangeable data structures
5. **Higher-Order Functions (HOFs)** — Functions as first-class values
6. **Functor (map)** — Structure-preserving transformations
7. **Applicative (ap / mapN)** — Applying wrapped functions
8. **Monad (flatMap / bind)** — Sequential context-aware computation

### Advanced Concepts (9-16)
9. **Natural Transformation** — Functor morphisms
10. **Monoid** — Associativity + identity
11. **Algebraic Data Types (ADTs) & Pattern Matching** — Sum/product types
12. **Effects at the Edges** — Pure core, impure boundaries
13. **Property-Based Testing** — Law verification
14. **Lazy Evaluation** — Deferred computation
15. **Tail Recursion / TCO** — Stack-safe recursion
16. **Language-Specific Features** — Type classes, generics, protocols, advanced types

---

## Build Systems & Dependencies

### Scala (SBT)
**Build file:** `build.sbt`
- **Version:** Scala 2.13.14
- **Dependencies:**
  - `cats-core` 2.12.0 (FP library with Functor, Applicative, Monad type classes)
  - `scalacheck` 1.17.0 (property-based testing, test scope)
- **Source directories:**
  - Main: `scala/`
  - Java interop: `java/`
- **Run commands:**
  ```bash
  sbt console                    # REPL
  :load scala/FPAnnotated.scala  # Load file
  scala scala/DemoMonadsImproved2.scala  # Run script
  ```

### Java (Maven)
**Build file:** `pom.xml`
- **Version:** Java 21 (with release flag)
- **Dependencies:**
  - `scala-library` 2.13.14 (for mixed Scala/Java builds)
  - `cats-core_2.13` 2.12.0
  - `junit-jupiter` 5.10.2
  - `jqwik` 1.8.4 (property-based testing)
- **Source directories:** `java/`, `scala/` (via build-helper-maven-plugin)
- **Encoding:** UTF-8 enforced (fixes Cp1252 errors)
- **Compilation order:** Scala first, then Java (allows Java to depend on Scala)
- **Run commands:**
  ```bash
  mvn clean compile              # Compile all sources
  mvn test                       # Run tests
  javac java/FPJavaAnnotated.java && java FPJavaAnnotated  # Direct run
  ```

### TypeScript
- **No package.json present** — Dependencies installed globally
- **Required packages:**
  ```bash
  npm install -g typescript ts-node
  npm install fast-check @types/node  # For testing
  ```
- **Run commands:**
  ```bash
  npx ts-node typescript/fpAnnotated.ts
  npx ts-node typescript/demoMonadsImproved2.ts
  ```

### Python
- **No requirements.txt** — Minimal dependencies
- **Version:** Python 3.10+ (requires structural pattern matching)
- **Optional packages:**
  ```bash
  pip install hypothesis dataclasses-json  # For testing
  ```
- **Run commands:**
  ```bash
  python3 python/fp_annotated.py
  python3 python/demo_monads_improved2.py
  ```

### JavaScript
- **Pure ES6+** — No build system required
- **Run command:**
  ```bash
  node javascript/fpAnnotated.js
  ```

---

## Code Conventions & Patterns

### Naming Conventions
- **Files:**
  - Main guides: `FPAnnotated.scala`, `FPJavaAnnotated.java`, `fp_annotated.py`, `fpAnnotated.ts`
  - Monad demos: `DemoMonads.scala`, `DemoMonadsImproved1.scala`, `DemoMonadsImproved2.scala`
  - Python: snake_case (`demo_monads.py`)
  - Others: PascalCase for classes, camelCase for files

- **Concepts:**
  - Functions/methods use descriptive names: `compose`, `inc`, `double`, `add`
  - Monadic operations: `map`, `flatMap`, `ap`, `bind`
  - Examples numbered sequentially (concept1, concept2, etc.)

### Code Style
1. **Extensive inline comments:** Every concept has multi-line explanatory comments
2. **Type signatures:** Explicitly typed where it aids understanding
3. **Step-by-step examples:** Code builds progressively from simple to complex
4. **Mathematical notation:** Comments reference λ-calculus (λx. x + 1)
5. **Law verification:** Identity, composition, associativity laws included
6. **Real-world examples:** Practical use cases alongside theory

### Documentation Pattern
Each implementation file follows this structure:
```
/*
 FP CONCEPTS (Language) — COMPREHENSIVE ANNOTATED GUIDE

 CORE CONCEPTS (1-8): [list]
 ADVANCED CONCEPTS (9-16): [list]
*/

// 1) Concept Name
code example
/*
Detailed explanation:
- What the code does
- Type breakdown
- Usage examples
- Key insights
- Common pitfalls
*/

// 2) Next Concept
...
```

---

## Development Workflow

### Adding New Concepts
1. **Multi-language implementation required:** Add concept to all 5 languages
2. **Maintain consistency:** Keep concept numbering and structure aligned
3. **Update README.md:** Document new concepts in the overview
4. **Include comments:** Follow existing annotation style
5. **Add examples:** Show both theory and practical usage

### Testing Approach
- **Property-based testing** for mathematical laws (ScalaCheck, jqwik, Hypothesis, fast-check)
- **Laws verified:** Functor laws, Monad laws, Monoid laws
- **No unit test directories currently** — examples are self-documenting
- Tests are embedded in main files using testing frameworks

### Git Workflow
- **Branch naming:** `claude/claude-md-<session-id>` for AI assistant work
- **Commit messages:** Descriptive, focus on concept added/improved
- **Recent pattern:** "Update.", "Concepts added.", "Adding Monads Demo improved."
- **No PR templates** — Direct commits to working branches

### Language-Specific Notes

#### Scala
- Uses **Cats library** extensively for type class instances
- Demonstrates **for-comprehensions** for monadic composition
- Shows **implicit evidence** for type class constraints
- Includes **tail recursion** with `@tailrec` annotation
- **LazyList** for lazy evaluation examples

#### Java
- Requires **Java 17+** for records and sealed classes
- Uses **functional interfaces:** `Function`, `BiFunction`, `Supplier`
- Demonstrates **Stream API** for lazy sequences
- Shows **trampoline pattern** for tail call optimization
- **Pattern matching** with switch expressions (Java 21)

#### Python
- Requires **Python 3.10+** for structural pattern matching
- Uses **dataclasses** for immutable types
- Demonstrates **protocols** (structural typing)
- Shows **generators** for lazy evaluation
- **Type hints** throughout for clarity

#### TypeScript
- Uses **advanced type system:** conditional types, mapped types, type-level programming
- Demonstrates **generator functions** for lazy evaluation
- Shows **class-based** and **function-based** patterns
- **Strict typing** with explicit generics
- No compilation step for development (ts-node)

#### JavaScript
- **Modern ES6+** syntax (arrow functions, destructuring)
- **No TypeScript overhead** for quick examples
- Demonstrates **closures** and **higher-order functions**
- Shows **generator functions** for lazy sequences

---

## Key Dependencies

### Cats (Scala FP Library)
- **Purpose:** Type class instances (Functor, Applicative, Monad, Monoid)
- **Version:** 2.12.0 (Scala 2.13)
- **Usage:** `import cats._`, `import cats.implicits._`
- **Key types:** `Option`, `Either`, `Validated`, `State`, `Reader`, `Writer`

### jqwik (Java Property Testing)
- **Purpose:** Property-based testing of mathematical laws
- **Version:** 1.8.4
- **Usage:** JUnit Platform integration, auto-discovery
- **Annotations:** `@Property`, `@ForAll`, `@Provide`

### Hypothesis (Python Property Testing)
- **Purpose:** Property-based testing
- **Usage:** `@given(st.integers())`

### fast-check (TypeScript Property Testing)
- **Purpose:** Property-based testing
- **Usage:** `fc.property(fc.integer(), ...)`

---

## Common AI Assistant Tasks

### Code Analysis
When analyzing code:
1. **Identify the concept number** (1-16) from comments
2. **Check cross-language consistency** — concept should be similar across languages
3. **Verify mathematical laws** — look for identity, composition, associativity
4. **Review type signatures** — ensure they match FP theory
5. **Check for purity** — no side effects in core logic

### Adding Examples
When adding new examples:
1. **Choose appropriate language** — consider existing patterns
2. **Follow annotation style** — extensive comments with step-by-step explanation
3. **Include type signatures** — explicit types for clarity
4. **Add usage examples** — show how to call the code
5. **Verify laws** — include property-based tests if applicable
6. **Update all languages** — maintain parity across implementations

### Refactoring
When refactoring:
1. **Preserve concept structure** — keep 1-16 ordering
2. **Maintain comments** — update explanations to match code
3. **Keep cross-language alignment** — similar structure across all languages
4. **Test property laws** — ensure refactoring doesn't break mathematical properties
5. **Update README.md** if concepts change

### Debugging
Common issues:
1. **Scala compilation:**
   - Missing Cats import: add `import cats.implicits._`
   - Type class instance not found: ensure Cats dependency in build.sbt

2. **Java compilation:**
   - Encoding errors: verify UTF-8 in pom.xml
   - Mixed Scala/Java: ensure scala-maven-plugin runs before maven-compiler-plugin

3. **TypeScript:**
   - Type errors: check generic constraints and variance
   - Missing types: install `@types/node`

4. **Python:**
   - Pattern matching errors: requires Python 3.10+
   - Protocol issues: use `typing.Protocol` for structural typing

### Documentation Updates
When updating docs:
1. **README.md** — User-facing, high-level overview
2. **CLAUDE.md** — AI assistant guide (this file)
3. **Inline comments** — Concept explanations in code
4. **File headers** — Concept summaries at top of each file

---

## File Navigation Tips

### Finding Concepts
- **Concept N in Scala:** `scala/FPAnnotated.scala` — search for `// N)`
- **Concept N in Java:** `java/FPJavaAnnotated.java` — search for `// N)`
- **Concept N in Python:** `python/fp_annotated.py` — search for `# N)`
- **Concept N in TypeScript:** `typescript/fpAnnotated.ts` — search for `// N)`

### Monad Examples
- **Basic:** `DemoMonads.*` files
- **Improved v1:** `DemoMonadsImproved1.*` files
- **Improved v2:** `DemoMonadsImproved2.*` files (most recent)
- **Modern patterns:** `DemoMonadsModern*.java`, `DemoMonadsLegacy*.java`

### Lambda Calculus
- Pure λ-calculus demonstrations in `LambdaCalculus.*` files across languages
- Shows Church encodings, combinators, Y-combinator

---

## Testing Approach

### Property-Based Testing
Each language uses property-based testing for law verification:

**Functor Laws:**
```
map(id) === id                    // Identity
map(f . g) === map(f) . map(g)    // Composition
```

**Monad Laws:**
```
flatMap(pure) === id              // Right identity
pure(a).flatMap(f) === f(a)       // Left identity
m.flatMap(f).flatMap(g) === m.flatMap(x => f(x).flatMap(g))  // Associativity
```

**Monoid Laws:**
```
x <> empty === x                  // Right identity
empty <> x === x                  // Left identity
(x <> y) <> z === x <> (y <> z)   // Associativity
```

### Running Tests
- **Scala:** ScalaCheck tests (if present in test/)
- **Java:** `mvn test` (jqwik auto-discovered)
- **Python:** `python -m pytest` or run file directly
- **TypeScript:** `npx jest` or run with ts-node

---

## Performance Considerations

### Stack Safety
- **Tail recursion:** Use `@tailrec` (Scala), trampoline pattern (Java), iteration (Python/TS)
- **Lazy evaluation:** `LazyList` (Scala), `Stream` (Java), generators (Python/TS/JS)
- **Large datasets:** Prefer lazy sequences over eager collections

### Memory Efficiency
- **Immutability:** Structural sharing reduces copying overhead
- **Stream processing:** Process data in chunks, don't materialize entire collections
- **Memoization:** Cache expensive computations (shown in some examples)

---

## Related Concepts

### Type Classes (Scala)
Demonstrated through Cats library:
- `Functor[F]` — map operation
- `Applicative[F]` — ap operation, pure
- `Monad[F]` — flatMap, pure
- `Monoid[A]` — combine, empty

### Sealed Types
- **Scala:** `sealed trait` for ADTs
- **Java 17+:** `sealed interface`/`sealed class`
- **Python:** Protocols and Union types
- **TypeScript:** Discriminated unions

### Pattern Matching
- **Scala:** Native `match` expression
- **Java 21:** Enhanced `switch` with patterns
- **Python 3.10+:** `match`/`case` statements
- **TypeScript:** Type narrowing with discriminated unions

---

## Contributing Guidelines

When contributing to this repository:

1. **Maintain cross-language parity** — If adding a concept to one language, add it to all
2. **Follow annotation style** — Extensive inline documentation is expected
3. **Verify laws** — Include property-based tests for algebraic structures
4. **Update README.md** — Keep user documentation in sync
5. **Keep examples self-contained** — Files should run independently
6. **Use consistent naming** — Follow established patterns
7. **Document dependencies** — Update build files (build.sbt, pom.xml) if adding libraries

### Suggested Improvements
From README.md contributing section:
- Additional concepts (Comonads, Free Monads, Optics/Lenses)
- More languages (Haskell, F#, Clojure, Rust, Kotlin)
- Performance benchmarks and optimizations
- Interactive examples and tutorials
- More comprehensive property-based test suites
- Separate test directories with extensive law verification

---

## Quick Reference

### Run All Examples
```bash
# Scala
sbt console
:load scala/FPAnnotated.scala

# Java
mvn clean compile
javac java/FPJavaAnnotated.java && java FPJavaAnnotated

# Python
python3 python/fp_annotated.py

# TypeScript
npx ts-node typescript/fpAnnotated.ts

# JavaScript
node javascript/fpAnnotated.js
```

### Build Commands
```bash
# Scala
sbt compile

# Java + Scala (Maven)
mvn clean compile
mvn test

# TypeScript (if tsconfig.json existed)
tsc

# Python (check syntax)
python3 -m py_compile python/*.py
```

### Clean Build Artifacts
```bash
# Scala
sbt clean

# Maven
mvn clean

# Manual cleanup
rm -rf target/ *.class
```

---

## Project Metadata

- **License:** MIT
- **Primary Language:** Scala (with multi-language examples)
- **Maintainer:** Functional Programming Community
- **Purpose:** Educational reference implementation
- **Status:** Active development (recent improvements to monadic examples)
- **Target Audience:** Developers learning FP across different languages

---

## AI Assistant Guidelines

### Best Practices
1. **Respect the structure** — 16 concepts, consistent across languages
2. **Maintain documentation quality** — Inline comments are critical
3. **Cross-language awareness** — Changes should be portable
4. **Preserve mathematical rigor** — Laws must hold
5. **Keep examples runnable** — Files should be self-contained
6. **Follow established patterns** — Don't introduce inconsistent styles

### When Making Changes
1. **Read existing code first** — Understand current patterns
2. **Check all language versions** — Maintain consistency
3. **Update comments** — Keep documentation in sync
4. **Verify compilation** — Test in relevant environment
5. **Update this file (CLAUDE.md)** if adding new conventions

### Communication Style
- **Explain FP concepts** using existing annotation style
- **Reference concept numbers** (1-16) when discussing features
- **Use mathematical notation** when appropriate (λ, ∘, etc.)
- **Cite language-specific features** (type classes, sealed types, protocols)
- **Link to relevant files** using relative paths

---

## Troubleshooting

### Scala Issues
**Problem:** `object cats is not a member of package`
**Solution:** Add Cats dependency to build.sbt, run `sbt update`

**Problem:** Tail recursion not optimized
**Solution:** Ensure `@tailrec` annotation, check function is actually tail-recursive

### Java Issues
**Problem:** `error: unmappable character (0xE2) for encoding Cp1252`
**Solution:** Verify UTF-8 encoding in pom.xml properties and compiler configuration

**Problem:** Scala classes not found in Java
**Solution:** Check scala-maven-plugin executes before maven-compiler-plugin

### Python Issues
**Problem:** `SyntaxError: invalid syntax` on match statement
**Solution:** Upgrade to Python 3.10+

**Problem:** Type hints not working
**Solution:** Install typing_extensions: `pip install typing_extensions`

### TypeScript Issues
**Problem:** Type inference fails on composed functions
**Solution:** Add explicit type parameters: `compose<A, B, C>(...)`

**Problem:** Module resolution errors
**Solution:** Install @types/node: `npm install @types/node`

---

## Additional Resources

### FP Theory
- **Bartosz Milewski's Category Theory for Programmers** — Foundational concepts
- **Scala with Cats book** — Cats library deep dive
- **Functional Programming in Scala (Red Book)** — Scala FP patterns

### Language-Specific
- **Scala:** Cats documentation, Scaladoc
- **Java:** Modern Java in Action, Functional interfaces JavaDoc
- **Python:** Functional Programming in Python, typing module docs
- **TypeScript:** TypeScript Handbook (Advanced Types section)

### Property-Based Testing
- **ScalaCheck User Guide** — Scala property testing
- **jqwik User Guide** — Java property testing
- **Hypothesis Documentation** — Python property testing
- **fast-check Documentation** — TypeScript/JavaScript property testing

---

## Version History

This documentation reflects the repository state as of the most recent commits:
- Enhanced monadic examples with fluent APIs
- Comprehensive multi-language coverage
- Advanced concepts including lazy evaluation and tail recursion
- Property-based testing framework integration
- Performance optimizations and stack-safe patterns

**Last Updated:** 2025-11-13
**For:** Claude AI Assistant usage
**Maintained by:** AI assistants working with this codebase
