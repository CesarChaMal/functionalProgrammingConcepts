# Functional Programming Concepts — Multi-language Examples

This project provides a **comprehensive, multi-language reference** for 13 fundamental **Functional Programming (FP)** concepts, fully implemented and documented in **Scala**, **Java**, **Python**, and **TypeScript**.

Each file contains **complete code examples** with in-line explanations, covering the theory, practical usage, and relevant laws.

---

## 📚 Concepts Included

1. **Lambda, Application, Currying, Partial Application** — Anonymous functions, function invocation, transforming multi-parameter functions into single-parameter chains
2. **Composition (∘)** — Combining functions where output of one becomes input of another
3. **Referential Transparency** — Expressions can be replaced with their values without changing program behavior
4. **Immutability** — Data structures that cannot be modified after creation
5. **Higher‑Order Functions (HOFs)** — Functions that take other functions as parameters or return functions
6. **Functor (map)** — Containers that can apply functions to wrapped values while preserving structure
7. **Applicative (ap / mapN)** — Enhanced functors that can apply wrapped functions to wrapped values
8. **Monad (flatMap / bind)** — Containers supporting sequential computation with context-aware chaining
9. **Natural Transformation** — Structure-preserving mappings between functors
10. **Monoid (associative op + identity)** — Types with associative binary operation and identity element
11. **Algebraic Data Types (ADTs) & Pattern Matching** — Sum and product types with exhaustive case analysis
12. **Effects at the Edges** — Isolating side effects to program boundaries while keeping core logic pure
13. **Property‑Based Testing of Laws** — Verifying mathematical properties hold across generated test cases

---

## 📂 Structure

```
/scala/FPAnnotated.scala       # Scala examples (uses Cats for Applicative/Monoid)
/java/FPJavaAnnotated.java     # Java examples (Optional, records, sealed interfaces)
/python/fp_annotated.py        # Python examples (dataclasses, match, Maybe)
/typescript/fpAnnotated.ts     # TypeScript examples (Option, discriminated unions, Promise)
README.md                      # Project overview and usage
```

---

## 🚀 Running the Code

### Scala

```bash
sbt console
:load scala/FPAnnotated.scala
```

Requires adding:

```scala
libraryDependencies += "org.typelevel" %% "cats-core" % "2.12.0"
```

### Java

```bash
javac java/FPJavaAnnotated.java
java FPJavaAnnotated
```

### Python

```bash
python3 python/fp_annotated.py
```

### TypeScript

```bash
npm install -g typescript ts-node
npx ts-node typescript/fpAnnotated.ts
```

---

## 🧠 Goals

* Understand FP concepts across multiple languages.
* Learn to apply theory (functors, applicatives, monads) in real code.
* Recognize shared patterns and idioms across ecosystems.
* Practice reasoning about laws like identity, composition, and associativity.

---

## 📜 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Open an issue for discussion before submitting pull requests.

**Author:** *Your Name*
