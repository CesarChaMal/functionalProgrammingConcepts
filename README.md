# Functional Programming Concepts — Multi-language Examples

This project provides a **comprehensive, multi-language reference** for **16 fundamental Functional Programming (FP) concepts**, fully implemented and documented in **Scala**, **Java**, **Python**, and **TypeScript**.

Each file contains **complete code examples** with in-line explanations, covering theory, practical usage, mathematical laws, and real-world applications.

---

## 📚 Core Concepts (1-8)

1. **Lambda, Application, Currying, Partial Application** — Anonymous functions, function invocation, transforming multi-parameter functions into single-parameter chains
2. **Composition (∘)** — Combining functions where output of one becomes input of another
3. **Referential Transparency** — Expressions can be replaced with their values without changing program behavior
4. **Immutability** — Data structures that cannot be modified after creation
5. **Higher‑Order Functions (HOFs)** — Functions that take other functions as parameters or return functions
6. **Functor (map)** — Containers that can apply functions to wrapped values while preserving structure
7. **Applicative (ap / mapN)** — Enhanced functors that can apply wrapped functions to wrapped values
8. **Monad (flatMap / bind)** — Containers supporting sequential computation with context-aware chaining

## 🚀 Advanced Concepts (9-16)

9. **Natural Transformation** — Structure-preserving mappings between functors
10. **Monoid (associative op + identity)** — Types with associative binary operation and identity element
11. **Algebraic Data Types (ADTs) & Pattern Matching** — Sum and product types with exhaustive case analysis
12. **Effects at the Edges** — Isolating side effects to program boundaries while keeping core logic pure
13. **Property‑Based Testing of Laws** — Verifying mathematical properties hold across generated test cases
14. **Lazy Evaluation** — Deferring computation until values are actually needed
15. **Tail Recursion / TCO** — Stack-safe recursive functions and optimization techniques
16. **Language-Specific Advanced Features** — Type classes (Scala), generics (Java), protocols (Python), advanced types (TypeScript)

---

## 📂 Structure

```
/scala/FPAnnotated.scala       # Scala examples (Cats, LazyList, type classes, tail recursion)
/java/FPJavaAnnotated.java     # Java examples (records, sealed classes, streams, trampolines)
/python/fp_annotated.py        # Python examples (dataclasses, generators, protocols, match)
/typescript/fpAnnotated.ts     # TypeScript examples (advanced types, generators, conditional types)
/scala/DemoMonads*.scala       # Monadic examples with fluent APIs
/java/DemoMonads*.java         # Java monadic patterns with records and streams
/python/demo_monads*.py        # Python monadic patterns with dataclasses
/typescript/demoMonads*.ts     # TypeScript monadic patterns with classes
README.md                      # Project overview and usage
```

---

## 🚀 Running the Code

### Scala

```bash
# Setup
sbt console
:load scala/FPAnnotated.scala

# Run monadic examples
scala scala/DemoMonadsImproved2.scala
```

**Dependencies** (build.sbt):
```scala
libraryDependencies ++= Seq(
  "org.typelevel" %% "cats-core" % "2.12.0",
  "org.scalacheck" %% "scalacheck" % "1.17.0" % Test
)
```

### Java (17+)

```bash
# Compile and run
javac java/FPJavaAnnotated.java
java FPJavaAnnotated

# Run monadic examples
javac java/DemoMonadsImproved2.java
java DemoMonadsImproved2
```

**Testing** (Maven):
```xml
<dependency>
    <groupId>net.jqwik</groupId>
    <artifactId>jqwik</artifactId>
    <version>1.7.4</version>
    <scope>test</scope>
</dependency>
```

### Python (3.10+)

```bash
# Run examples
python3 python/fp_annotated.py
python3 python/demo_monads_improved2.py

# Install testing dependencies
pip install hypothesis dataclasses-json
```

### TypeScript

```bash
# Setup
npm install -g typescript ts-node

# Run examples
npx ts-node typescript/fpAnnotated.ts
npx ts-node typescript/demoMonadsImproved2.ts

# Install testing dependencies
npm install fast-check @types/node
```

---

## 🧠 Goals

* **Cross-language Understanding**: Master FP concepts across different programming paradigms
* **Theory to Practice**: Apply mathematical concepts (functors, applicatives, monads) in production code
* **Pattern Recognition**: Identify shared FP idioms and abstractions across language ecosystems
* **Law-based Reasoning**: Understand and verify mathematical properties like identity, composition, and associativity
* **Performance Awareness**: Learn stack-safe recursion, lazy evaluation, and optimization techniques
* **Type Safety**: Leverage advanced type systems for compile-time correctness

## 🏆 Language Comparison Summary

| Language | **Strengths** | **Best For** | **FP Rating** |
|----------|---------------|--------------|---------------|
| **Scala** | Native FP, concise syntax, powerful type system | Learning FP theory, high-performance backends | ⭐⭐⭐⭐⭐ |
| **Java** | Enterprise ecosystem, strong tooling, performance | Production systems, large teams | ⭐⭐⭐⭐ |
| **TypeScript** | Modern syntax, web ecosystem, gradual typing | Web development, full-stack applications | ⭐⭐⭐⭐ |
| **Python** | Readable, approachable, rich libraries | Data science, rapid prototyping, education | ⭐⭐⭐ |

---

## 📊 Recent Improvements

- **Enhanced Monadic Examples**: Added fluent API implementations across all languages
- **Advanced Concepts**: Lazy evaluation, tail recursion, and language-specific features
- **Comprehensive Comments**: Detailed explanations with step-by-step traces
- **Performance Focus**: Stack-safe recursion and optimization techniques
- **Type Safety**: Advanced type system features and constraints

---

## 📜 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional FP concepts (Comonads, Free Monads, Optics)
- More language examples (Haskell, F#, Clojure, Rust)
- Performance benchmarks and optimizations
- Interactive examples and tutorials

Open an issue for discussion before submitting pull requests.

**Maintainer:** *Functional Programming Community*
