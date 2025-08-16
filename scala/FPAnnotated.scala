/*
 FP CONCEPTS (Scala) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — Lambdas are anonymous functions; application calls them.
    Currying converts (x,y) -> z into x -> (y -> z). Partial application fixes some args and returns a new func.
 2) Composition (∘) — If g: A->B and f: B->C then (f ∘ g): A->C; promotes pipeline thinking.
 3) Referential Transparency — Replace an expression by its value without changing behavior (pure, side‑effect free).
 4) Immutability — Create new values instead of mutating; avoids shared‑state bugs.
 5) Higher‑Order Functions (HOFs) — Functions that take/return functions; map/filter/reduce pipelines.
 6) Functor — Context F[_] with map that preserves structure and laws (identity, composition).
 7) Applicative — Combine independent contexts; e.g., building values from multiple Options.
 8) Monad — flatMap to sequence dependent computations; laws: left/right identity, associativity.
 9) Natural Transformation — Uniform, structure‑preserving F[A] -> G[A]; e.g., List ~> Option.
 10) Monoid — Associative binary op with identity; enables safe folding.
 11) ADTs & Pattern Matching — Sum/Product types with exhaustive handling.
 12) Effects at the Edges — Keep core logic pure; do I/O at boundaries.
 13) Property‑Based Testing (outline) — Check laws via randomized tests (ScalaCheck/Cats‑laws).

 Note: For Applicative/Monoid helpers we use Cats. In sbt add:
 libraryDependencies += "org.typelevel" %% "cats-core" % "2.12.0"
*/

import cats._
import cats.implicits._
import scala.Function.const

object FPAnnotated extends App {
  // 1) Lambda, Application, Currying, Partial Application
  val inc: Int => Int = x => x + 1                  // λx. x + 1
  val add: Int => Int => Int = x => y => x + y      // λx. λy. x + y (curried)
  val add5: Int => Int = add(5)                     // partial application
  val seven: Int = add5(2)                          // application => 7
/*
  1. Line 1: a simple lambda (a function value)

  - Code: val inc: Int => Int = x => x + 1
  - Meaning:
    - inc is a function from Int to Int.
    - x => x + 1 is an anonymous function (lambda) that increments its argument.

    - Usage:
    - inc(10) == 11

  - Notes:
    - Pure: same input → same output; no side effects.
    - In Scala, calling a function is just inc(10) (apply is invoked implicitly).

  2. Line 2: a curried function (a function that returns a function)

  - Code: val add: Int => Int => Int = x => y => x + y
  - Meaning:
    - add takes one Int (x) and returns another function Int => Int that takes y and computes x + y.
    - This is “curried” form: x => (y => x + y).

  - Why this is useful:
    - Enables partial application: you can fix the first argument and get a specialized function.
    - Plays nicely with composition and higher-order functions.

  - Closure:
    - The inner function y => x + y “remembers” x (it closes over x).

  3. Line 3: partial application (fix the first argument)

  - Code: val add5: Int => Int = add(5)
  - Meaning:
    - Provide the first argument x = 5 to add. You get back a function y => 5 + y.
    - Type now is Int => Int (a one-argument function).

    - Usage:
    - add5(10) == 15

  4. Line 4: function application (call the specialized function)

  - Code: val seven: Int = add5(2)
  - Meaning:
    - Call add5 with 2 → computes 5 + 2 = 7.
  - seven == 7

  Equivalences and tips
  - Direct two-step application:
    - add(5)(2) == 7

  - Uncurried vs curried:
    - Uncurried form would be (Int, Int) => Int, e.g., val add2: (Int, Int) => Int = (x, y) => x + y
  - In Scala, you can convert between curried/uncurried with Function.uncurried and Function.curried if needed.

    - Type inference:
    - You could write val inc: Int => Int = _ + 1 to be more concise.
*/


  // 2) Composition (∘): compose f after g to avoid temporaries
  def compose[A,B,C](f: B => C, g: A => B): A => C = a => f(g(a))
  val double: Int => Int = _ * 2
  val compRes: Int = compose(inc, double)(10)       // 21 = inc(double(10))
/*
  1. Line 1: define a generic compose

    - Code: def compose[A,B,C](f: B => C, g: A => B): A => C = a => f(g(a))
  - Meaning:
    - Given two functions g: A => B and f: B => C, compose returns a new function h: A => C such that h(a) = f(g(a)).
    - Type parameters A, B, C make it fully generic over input and output types.

  - Why it’s useful:
    - Builds pipelines without intermediate variables: “do g first, then f”.

  2. Line 2: define a “double” function

  - Code: val double: Int => Int = _ * 2
  - Meaning:
    - double is a function taking an Int and returning that Int times 2.

  - Usage:
    - double(7) == 14

  3. Line 3: apply the composed function

    - Code: val compRes: Int = compose(inc, double)(10)
  - Step-by-step trace:
    1. compose(inc, double) creates a new function h(a) = inc(double(a)).
  2. Call h(10):
    - double(10) = 20
  - inc(20) = 21

  3. Result is 21, so compRes == 21.

  Extra notes
    - Order matters:
    - compose(f, g)(x) = f(g(x)) means “run g first, then f”.

  - Types in this instance:
    - double: Int => Int
  - inc: Int => Int
  - compose(inc, double): Int => Int

  - Mental model:
    - Composition wires small functions together: output of one feeds the next, keeping code declarative and avoiding temporaries.
*/

  // 3) Referential Transparency: pure expressions can be replaced with their values
  val pureTotal: Int = List(1,2,3).sum              // == 6; safe replacement
  // println("hi") // I/O is a side effect → not referentially transparent
/*
  1. What “referential transparency” means

  - An expression is referentially transparent if you can replace it with its value everywhere in the program without changing the program’s behavior.
  - In practice: no side effects, no hidden/external state, deterministic (same input → same output).

  2. The pure expression

  - Code: List(1,2,3).sum
  - Meaning:
    - Constructs a List(1, 2, 3).
  - sum folds the elements with addition starting from the identity 0.

  - Micro-trace:
    - Start: 0
  - 0 + 1 = 1
  - 1 + 2 = 3
  - 3 + 3 = 6
  - Result: 6

  - Why it’s referentially transparent:
    - No I/O, no mutation, no randomness.
  - You can safely replace List(1,2,3).sum with 6 anywhere. The program’s behavior won’t change.

  - Therefore: pureTotal is 6, and writing val pureTotal = 6 is behaviorally equivalent.

  3. The side‑effecting expression (not referentially transparent)

  - Code: println("hi")
  - Why it’s not referentially transparent:
    - It performs I/O by printing to stdout.
  - Replacing it with a “value” (it returns Unit) would remove the print, changing observable behavior.
    - So it cannot be freely substituted by its “value” without altering the program.

  Key takeaways
    - Pure expressions are interchangeable with their values—great for reasoning, testing, and refactoring.
  - Side effects (like printing) are not referentially transparent; they change the outside world and cannot be replaced by a value without changing behavior.
*/

  // 4) Immutability: new values, no mutation
  final case class Point(x: Int, y: Int)
  val p1 = Point(1,1)
  val p2 = p1.copy(x = 2)                           // p1 unchanged
/*
  1. Line 1: define an immutable data carrier (a case class)

  - Code: final case class Point(x: Int, y: Int)
  - What a case class is:
  - A concise way to define immutable, value‑semantics data.
  - Automatically provides:
    - A companion with an apply constructor: Point(x: Int, y: Int)
  - equals, hashCode, and toString based on fields
  - Product/tuple‑like behavior (e.g., pattern matching support, unapply)

  - Fields in a case class are vals by default (read‑only).

  - Immutability here:
    - You cannot reassign Point’s fields after construction.
  - Note: immutability is shallow—if a field holds a mutable object, the case class won’t freeze it. Using Ints here is fully safe.

  - final:
  - Prevents subclassing, keeping the data type closed and predictable.

  2. Line 2: create the first instance

    - Code: val p1 = Point(1,1)
  - Meaning:
    - Uses the generated apply constructor to build Point(1, 1).
    - p1 is a value (val) referencing an immutable Point.
  - Field access is direct: p1.x == 1, p1.y == 1.

  3. Line 3: create a new instance instead of mutating

  - Code: val p2 = p1.copy(x = 2)
  - Step‑by‑step:
    1. Read p1.y (implicitly used by copy since we don’t override y) → 1 (read, not mutate).
  2. copy creates a new Point where x is overridden to 2, y is carried over: Point(2, 1).
  3. Bind this new value to p2.

    - Key idea:
    - You didn’t “change” p1; you constructed a new value p2 with a modified field.
    - p1 remains Point(1, 1). p2 is Point(2, 1).

    Why this matters in FP
    - Immutability avoids accidental shared‑state bugs.
  - Values are safe to pass around and reuse; reasoning is simpler (no hidden changes).
    - Plays well with parallelism and caching.
    - When you need a “modified” version, you build a new value with copy (as shown with p2).
*/

  // 5) Higher‑Order Functions: map/filter/reduce composition
  val hofSum: Int = List(1,2,3).map(_ * 2).filter(_ > 2).reduce(_ + _) // 10
  /*
  1. Built-in HOF pipeline (map → filter → reduce)
  - What happens:
    - Start with List(1, 2, 3)
    - map(_ * 2)   → List(2, 4, 6)
    - filter(_ > 2)→ List(4, 6)
    - reduce(_ + _)→ 10

  - Why these are HOFs:
    - map, filter, reduce all receive functions as parameters (lambdas), so they’re higher-order functions.
  */

  // Custom HOF #1: map for List
  def map[A, B](xs: List[A])(f: A => B): List[B] = {
    // Implemented manually to show the idea (don’t use xs.map)
    xs.foldRight(List.empty[B])((a, acc) => f(a) :: acc)
  }
  // Usage: doubles each element -> List(2, 4, 6)
  val customDoubled: List[Int] = map(List(1,2,3))(_ * 2)
  /*
2. Custom HOF #1: map for List

- Purpose:
  - Transform each element of a list independently using a provided function, producing a new list (possibly of a different element type).

- Signature reasoning:
  - map[A, B](xs: List[A])(f: A => B): List[B]
  - A and B are type parameters. xs is your input list; f is the behavior (A -> B) to apply to each element.
  - Curried form (xs)(f) makes it easy to partially apply xs or pass map(List(...)) around.

- Behavior (step-by-step):
  - Use foldRight to traverse xs from right to left, building a new list.
  - For each element a:
    - Compute b = f(a)
    - Prepend b to the accumulator: b :: acc
  - Return the accumulated list.

- Example:
  - map(List(1,2,3))(_ * 2) → List(2, 4, 6)
  - map(List("a","bb"))(_.length) → List(1, 2)

- Complexity:
  - Time O(n), Space O(n) for the new list, where n = xs.length.

- Purity and referential transparency:
  - If f is pure, map is pure; output depends only on xs and f with no side effects.

- Why it’s higher‑order:
  - It takes a function f as an argument.

- Variations:
  - Implement with foldLeft + reverse (trade-offs in allocation/stack use).
  - Parallel mapping for pure f (e.g., with parallel collections).

- Edge cases:
  - Nil → Nil (empty input yields empty output).
  - Avoid side effects in f to preserve reasoning and testability.
*/

  // Custom HOF #2: map -> filter -> reduce as one reusable helper
  def mapFilterReduce[A, B](
                             xs: List[A]
                           )(
                             mapper: A => A,
                             predicate: A => Boolean,
                             zero: B,
                             reducer: (B, A) => B
                           ): B = {
    xs.foldLeft(zero) { (acc, a) =>
      val m = mapper(a)
      if (predicate(m)) reducer(acc, m) else acc
    }
  }
  // Usage: same logic as the pipeline above -> 10
  val customMfr: Int = mapFilterReduce(List(1,2,3))(
    mapper    = _ * 2,   // List(2,4,6)
    predicate = _ > 2,   // List(4,6)
    zero      = 0,
    reducer   = (acc: Int, a: Int) => acc + a    // make parameter types explicit
  )
  /*
  3. Custom HOF #2: mapFilterReduce (one reusable helper)

  - Purpose:
    - Perform a “map → filter → reduce” pipeline in a single pass, avoiding intermediate collections and improving performance.

  - Signature reasoning:
    - mapFilterReduce[A, B](xs: List[A])(mapper: A => A, predicate: A => Boolean, zero: B, reducer: (B, A) => B): B
    - mapper is A => A to keep a single element type flowing; you can generalize to mapper: A => C with extra type params and adjust predicate/reducer accordingly.
    - zero is the initial accumulator; reducer folds an A into B.

  - Behavior (step-by-step):
    - Start with acc = zero.
    - For each a in xs:
      - m = mapper(a)          // map step
      - if predicate(m) then   // filter step
          acc = reducer(acc, m)// reduce step
    - Return acc.

  - Example:
    - xs = List(1,2,3), mapper = _ * 2 → List(2,4,6)
    - predicate = _ > 2 → keep List(4,6)
    - zero = 0, reducer = _ + _ → 0 + 4 + 6 = 10

  - Complexity:
    - One traversal: Time O(n), Space O(1) extra.

  - Purity and referential transparency:
    - If mapper, predicate, and reducer are pure, the whole function is pure and deterministic.

  - Why it’s higher‑order:
    - Accepts three behaviors: mapper, predicate, reducer.

  - When to use:
    - You want pipeline clarity with the efficiency of a single pass (no intermediate lists).

  - Variations:
    - Generalize mapper to change element type; allow early termination by returning a short‑circuiting accumulator (e.g., Either/Option wrapper).

  - Edge cases:
    - Empty list → returns zero.
    - predicate always false → returns zero.
    - Non‑associative reducers can be used, but for parallelization or law‑based reasoning prefer associative reducers with proper identities.
  */

  // Custom HOF #3: return a new function by repeating an operation n times
  def repeat[T](n: Int)(f: T => T): T => T = {
    // Compose f with itself n times
    (0 until n).foldLeft((x: T) => x)((acc, _) => acc.andThen(f))
  }
  // Usage: reuse an `inc` function, apply it 3 times
//  val inc: Int => Int = _ + 1
  val inc3: Int => Int = repeat(3)(inc)
  val inc3Res: Int = inc3(10) // 13
  /*
  4. Custom HOF #3: repeat (returns a new function)

  - Purpose:
    - Build a new function that applies f to its input n times:
      repeat(n)(f)(x) = f(f(...f(x)...)) with n applications.

  - Signature reasoning:
    - repeat[T](n: Int)(f: T => T): T => T
    - f must be an endofunction (T => T) so it can compose with itself.

  - Behavior (step-by-step):
    - Start with identity: (x: T) => x.
    - Compose f onto it n times using andThen.
    - Return the composed function.

  - Example:
    - inc = _ + 1; repeat(3)(inc)(10) → 13
    - repeat(3)((s: String) => s + "!")("go") → "go!!!"

  - Complexity:
    - Building the function: O(n) composition.
    - Calling the result: O(1) for the function reference (actual cost is the composed chain created once).

  - Purity and referential transparency:
    - If f is pure, repeat returns a pure function; same input → same output.

  - Why it’s higher‑order:
    - Takes a function and returns a new function.

  - Variations:
    - Alternative implementation that applies f n times at call‑site (looping each call) trades build‑time for call‑time.
    - n = 0 naturally yields identity.

  - Edge cases:
    - n = 0 → identity function.
    - Large n with expensive f: consider algebraic shortcuts (e.g., exponentiation‑by‑squaring for certain combinable transformations).

  Key takeaways
  - All these are higher-order because they take functions as parameters or return functions.
  - The built-in example uses standard library HOFs; the custom examples show how to build your own reusable functional utilities.
  - mapFilterReduce demonstrates how to package a common pipeline into a single, composable abstraction.
  - repeat shows building new behavior (n-fold application) by returning a composed function.
  */

  // 6) Functor: map preserves structure (laws: identity, composition)
  val fList = List(1,2,3).map(_ + 1)                // List(2,3,4)
  val fOpt  = Option(42).map(_ + 1)                 // Some(43)
  /*
  1. Lines: map over List and Option (Functor behavior)

  - Code:
    - List(1,2,3).map(_ + 1)  → List(2,3,4)
    - Option(42).map(_ + 1)   → Some(43)
  - Meaning:
    - map applies a function to each inner value while preserving the outer “shape”:
      - For List, length/order stay the same; elements are transformed.
      - For Option, Some(x) becomes Some(f(x)); None stays None.

  2. What “Functor” means (informal, Scala notation)
  - A type constructor F[_] that supports a structure‑preserving map:
    - def map[A,B](fa: F[A])(f: A => B): F[B]
  - Here:
    - F = List or Option.
    - Only the contents change; the container’s structure (size/presence) is preserved.

  3. Why map is useful
  - Eliminates manual case analysis:
    - Option: instead of matching on Some/None to transform, map does it safely.
    - List: declarative element-wise transformation without mutation.

  4. Behavior summary
  - Option:
    - Some(v).map(f)  == Some(f(v))
    - None.map(f)     == None
  - List:
    - xs.map(f) transforms each element; size == xs.size

  5. Functor laws (enable safe refactoring)
  - Identity:
    - fa.map(identity) == fa
  - Composition:
    - fa.map(g compose f) == fa.map(f).map(g)
  - With pure f and g these hold, allowing you to fuse or reorder maps.

  6. Purity and effects
  - If f is pure, map is referentially transparent:
    - Same input + same f → same output; no side effects.
  - Side effects inside f will run per element (List) or when present (Option).

  7. Edge cases and tips
  - Option:
    - Option(null) is Some(null); use Option when a value may be missing, not as a null wrapper.
  - List:
    - Empty List maps to empty List.
    - map is O(n) in time, allocates a new List.

  8. Chaining example
  - Option(10).map(_ + 1).map(_ * 2)   // Some(22)
  - List(1,2,3).map(_ * 2).map(_ + 1)  // List(3,5,7)
  */

  // 7) Applicative: combine independent Option values
  val name: Option[String] = Some("Ada")
  val age:  Option[Int]    = Some(36)
  val user: Option[(String, Int)] = (name, age).tupled           // Some(("Ada",36))
  val userStr: Option[String] = (name, age).mapN((n,a) => s"$n is $a")
  /*
  -------------------------------------------------------------------------------
  Explanation — Applicative combination with `tupled` and `mapN` (Cats)
  -------------------------------------------------------------------------------
  1) Lines and meaning
     - `val user = (name, age).tupled`
       • Uses the Applicative instance for `Option` to combine two independent Option values.
       • If both are `Some`, produce `Some((n, a))`; if either is `None`, result is `None`.

     - `val userStr = (name, age).mapN((n,a) => s"$n is $a")`
       • Like `tupled` but applies a function `(String, Int) => String` to the unwrapped values.
       • If both present: yields `Some("Ada is 36")`; otherwise `None`.

  2) Why “Applicative” (intuition)
     - Applicative composes **independent** effects/contexts (here, presence/absence in `Option`).
     - Signature shape (for 2 args): `map2: (A, B) => C, Option[A], Option[B] => Option[C]`.
     - No dependency between `name` and `age` is required; they are combined in one step.
     - Contrast with Monad: Monads (`flatMap`) handle **dependent** sequencing.

  3) How it works under the hood
     - `tupled` is equivalent to a `map2` that builds a pair: `map2(name, age)((n, a) => (n, a))`.
     - `mapN` is `map2` with your function: `map2(name, age)(f)`.
     - Rules:
       • If any input is `None` → short-circuit to `None` (no function run).
       • If all are `Some` → run the function once and wrap in `Some`.

  4) Typical use cases
     - Constructing DTOs/records from multiple optional fields.
     - Independent validation/combination. (For accumulating errors, prefer `Validated` over `Option`.)

  5) Purity & referential transparency
     - With a pure function, `mapN`/`tupled` are pure: same inputs → same output, no side effects.
     - The function is executed only when **all** inputs are present.

  6) Edge cases / tips
     - If values might be `null`, produce `Option` safely with `Option(x)` instead of `Some(null)`.
     - For 3+ args use `(a, b, c).mapN(f)` etc.; Cats provides arity up to 22.
     - `tupled` yields a tuple; you can `.map` over it later or use `mapN` directly to shape the output.

  7) Laws (Applicative intuition)
     - Homomorphism: `mapN(Some(a), Some(b))(f) == Some(f(a,b))`.
     - Interchange/Composition (informal for `Option`): combining is associative up to tuple restructuring when all are `Some`; `None` dominates.

  8) Chaining examples
     - `(name, age).tupled.map{ case (n, a) => s"$n is $a" } == (name, age).mapN((n,a) => s"$n is $a")`.
     - `(Some(10), Some(5)).mapN(_ + _) == Some(15)`; `(Some(10), None).mapN(_ + _) == None`.
  -------------------------------------------------------------------------------
  */

  // 8) Monad: sequence dependent computations with flatMap / for‑comprehension
  val monadRes: Option[Int] = for { x <- Option(2); y <- Option(3) } yield x + y   // Some(5)
  /*
  1. Line: sequence dependent Option computations using Scala's for‑comprehension (Monad behavior)

  - Code: for { x <- Option(2); y <- Option(3) } yield x + y
  - Meaning:
    - Desugar: Option(2).flatMap(x => Option(3).map(y => x + y))
    - Start with Option(2). If present, bind value to x.
    - Next, evaluate Option(3). If present, bind to y.
    - Finally, compute x + y and wrap the result in Some.
    - If either Option is None, the whole chain short‑circuits to None.

  - Result here:
    - x = 2; y = 3 → 2 + 3 = 5 → Some(5).

  2. Why “Monad” here
  - A monad provides flatMap/bind to chain dependent computations.
  - For Option:
    - flatMap ensures that if a step is None, all later steps are skipped.
    - This models computations that may fail or be absent.

  3. For‑comprehension syntax (Scala sugar)
  - for { x <- Option(2); y <- Option(3) } yield x + y
  - Equivalent to nesting flatMap/map:
    Option(2).flatMap(x => Option(3).map(y => x + y))
  - Cleaner for multiple steps.

  4. Behavior summary
  - Present case: Some(2), Some(3) → combine into Some(5).
  - Empty case: If either is None, result is None.
  - Short‑circuiting happens automatically.

  5. Typical sequencing patterns
  - Two steps: oa.flatMap(a => ob.map(b => f(a,b)))
  - Three steps: oa.flatMap(a => ob.flatMap(b => oc.map(c => g(a,b,c))))
  - For‑comprehensions scale neatly to many steps.

  6. Purity and transparency
  - If the inner functions are pure:
    - Same inputs → same outputs.
    - No hidden side effects.
  - Side effects in yield body only execute when all Options are present.

  7. Edge cases & tips
  - Avoid null: use Option(x) which becomes None if x is null.
  - Readability: for‑comprehensions are preferred over nested flatMaps.
  - Debugging: insert println/logging in yield body to see when it runs.

  8. Monad laws (with pure functions)
  - Left identity: Option(a).flatMap(f) == f(a)
  - Right identity: opt.flatMap(Option(_)) == opt
  - Associativity: opt.flatMap(f).flatMap(g) == opt.flatMap(a => f(a).flatMap(g))
    - These ensure predictable refactoring and correctness.
  */

  // 9) Natural Transformation: List ~> Option (headOption)
  def headOption[A](xs: List[A]): Option[A] = xs.headOption
  val nat: Option[Int] = headOption(List(10, 20))                   // Some(10)
  /*
  1. Line: a natural transformation between containers (List -> Option)

  - Code: headOption(xs)
  - Meaning:
    - Converts a List[A] into an Option[A] by taking its first element if present.
    - Scala's standard library already provides `.headOption` on List, which returns:
      - Some(first element) if the list is non-empty.
      - None if the list is empty.

  - Result examples:
    - headOption(List(10, 20)) → Some(10)
    - headOption(List())       → None

  2. What "Natural Transformation" means (informal)
  - A uniform, structure-preserving mapping between type constructors F and G:
    - For all A, a function nat: F[A] -> G[A] that does not depend on the specifics of A.
  - Here:
    - F = List, G = Option, nat = headOption
    - Uniform: same logic for integers, strings, or any other type.

  3. Why this is useful
  - It changes the container context without altering the element type.
  - Collapses the “many (possibly zero)” context of List into the “zero or one” context of Option.
  - Allows seamless composition with APIs that expect Option.

  4. Behavior and properties
  - headOption is total for all List[A]:
    - Empty → None
    - Non-empty → Some(first element)
  - Pure and referentially transparent:
    - Same input list → same output Option; no side effects.

  5. Edge cases and tips
  - Unlike Java's Optional example, Scala's List.headOption already handles null safety for empty lists.
  - If the first element itself is null (rare in idiomatic Scala), result will be Some(null).

  6. Variations
  - lastOption: Scala also provides `.lastOption` for the last element.
  - Safe head for arrays/iterables can be defined similarly.

  7. Composition intuition
  - Natural transformations compose:
    - For example, if you have Option[A] -> Either[E,A], you can compose with headOption to get List[A] -> Either[E,A].
  */

  // 10) Monoid: associative op + identity; Cats supplies instances
  val mSum: Int = List(1,2,3).combineAll                            // 6 (Int monoid is +/0)
  val mStr: String = List("a","b","c").combineAll                 // "abc" (concat/"")
  /*
  10) Monoid with Cats – associative operation + identity

  Code:
    val mSum: Int = List(1,2,3).combineAll
    val mStr: String = List("a","b","c").combineAll

  1. Line: reduce with a Monoid (via Cats typeclass instances)
  - combineAll comes from Cats and requires a Monoid instance for the element type.
  - Cats already provides:
    - Monoid[Int] where ⊕ = + and identity = 0
    - Monoid[String] where ⊕ = concat and identity = ""

  2. Meaning:
  - A Monoid is a pair (⊕, e) with:
    - Associative binary operation ⊕: (x ⊕ y) ⊕ z == x ⊕ (y ⊕ z)
    - Identity element e: e ⊕ x == x == x ⊕ e
  - combineAll folds a list using the given monoid:
    - List(1,2,3).combineAll = 1 + 2 + 3 with start 0 → 6
    - List("a","b","c").combineAll = "a" + "b" + "c" with start "" → "abc"

  3. Behavior (step-by-step)
  - mSum:
    acc = 0
    0 + 1 = 1
    1 + 2 = 3
    3 + 3 = 6
    result = 6

  - mStr:
    acc = ""
    "" + "a" = "a"
    "a" + "b" = "ab"
    "ab" + "c" = "abc"
    result = "abc"

  4. Why associativity and identity matter
  - Associativity:
    Needed so Cats can fold in any grouping, especially in parallel/distributed contexts.
    Example: ("a" + "b") + "c" == "a" + ("b" + "c") == "abc"
  - Identity:
    Ensures folding works on empty lists.
    Example: List[Int]().combineAll == 0
             List[String]().combineAll == ""

  5. Purity and determinism
  - combineAll is pure and referentially transparent:
    Same list → same result; no side effects.

  6. Edge cases and tips
  - Empty list returns the identity element automatically.
  - Be careful with non-associative operations; Cats won’t guarantee correctness.
  - Performance: concat is O(n^2) if used naively; Cats is fine for small lists, but for very large ones consider alternatives like StringBuilder.

  7. Custom Monoids
  - You can define your own:
    implicit val boolAndMonoid: Monoid[Boolean] = new Monoid[Boolean] {
      def empty = true
      def combine(x: Boolean, y: Boolean) = x && y
    }
    List(true, false, true).combineAll → false

  8. Parallel friendliness
  - Associativity allows safe folding in parallel collections and distributed systems.
  */

  // 11) ADTs & Pattern Matching: encode domain and handle exhaustively
  sealed trait Shape
  final case class Circle(r: Double) extends Shape
  final case class Rect(w: Double, h: Double) extends Shape
  def area(s: Shape): Double = s match {
    case Circle(r) => Math.PI * r * r
    case Rect(w,h) => w * h
  }
  /*
  1. Lines 1–3: define an Algebraic Data Type (ADT) with a sealed trait and case classes

  - Code:
    - sealed trait Shape
    - final case class Circle(r: Double) extends Shape
    - final case class Rect(w: Double, h: Double) extends Shape
  - Meaning:
    - A sealed trait limits which classes can extend it to the same compilation unit.
    - Case classes automatically provide immutability, structural equality, pattern matching support, and concise syntax.
    - Together they define a closed set of Shape variants (sum type): Shape = Circle | Rect.

  - Why sealed matters:
    - The compiler knows all possible subtypes, so pattern matching on Shape is checked for exhaustiveness.
    - Prevents external extensions, keeping the domain model closed and safe.

  2. Lines 5–8: pattern matching with match expression

  - Code:
    - s match { case Circle(r) => ...; case Rect(w,h) => ... }
  - Meaning:
    - Deconstructs the concrete Shape into its parameters (r, w, h).
    - Scala enforces exhaustive matching for sealed traits—no default needed.
    - Adding a new Shape variant will cause a compile‑time warning/error until all matches are updated.

  3. Behavior (step-by-step)

  - If s is Circle(r):
    - area = π * r * r
  - If s is Rect(w,h):
    - area = w * h

  4. Why ADTs + pattern matching are powerful

  - Clarity:
    - The domain is modeled with a finite, explicit set of cases.
  - Safety:
    - Exhaustive matching ensures all possibilities are handled at compile time.
  - Immutability:
    - Case classes are immutable by default, making reasoning easier.
  - Refactoring:
    - Adding a new Shape requires updates in all match expressions, guided by the compiler.

  5. Purity and reasoning

  - area is pure:
    - It depends only on its input, no side effects.
  - Referential transparency:
    - Example: area(Circle(2)) can be replaced with 12.566... without changing program behavior.

  6. Edge cases and tips

  - Negative/NaN parameters:
    - Not validated by default; validation logic can be added in companion objects or constructors.
  - Extensibility:
    - Adding Triangle requires modifying Shape and updating all pattern matches.
  - Performance:
    - Pattern matching is efficient (O(1) dispatch), and case classes are lightweight immutable data types.
  */

  // 12) Effects at the Edges: pure core + I/O boundary
  object Domain { def pureLogic(x: Int): Int = x * 2 }
  println(Domain.pureLogic(5)) // side effect only at boundary
  /*
  1. Lines 1–2: separate pure core from I/O

  - Code:
    - object Domain { def pureLogic(x: Int): Int = x * 2 }
    - println(Domain.pureLogic(5))
  - Meaning:
    - pureLogic: a pure function (deterministic, no I/O, no mutation).
    - println: side effect at boundary (writes to stdout).
    - This separation follows FP principle: keep domain logic pure, perform effects at edges.

  2. Why this separation matters

  - Testability:
    - pureLogic is trivial to unit test: same input always gives same output.
  - Reasoning & refactoring:
    - Pure code is referentially transparent; you can reason about and safely refactor it.
  - Reuse & composition:
    - Pure functions compose easily; you can build pipelines.
  - Reliability & observability:
    - Centralize side effects, making them easier to monitor and retry.

  3. Behavior (step‑by‑step)

  - Domain.pureLogic(5) → 10 (pure computation).
  - println(10) → prints "10" to console (side effect).

  4. Purity vs effects

  - pureLogic:
    - No global state, no I/O, deterministic; returns x * 2.
  - println:
    - Performs an I/O action; not referentially transparent.

  5. Edge cases and tips

  - Exceptions:
    - Should be handled at I/O boundary; core remains pure.
  - Time, randomness, config:
    - Pass them as parameters instead of embedding inside pure functions.
  - Nulls:
    - Prefer Option types in Scala to model absence safely.

  6. Testing strategy

  - Unit tests:
    - assert(Domain.pureLogic(5) == 10)
  - Integration tests:
    - Verify console/log output at boundary.

  7. Refactoring patterns

  - Dependency injection at edges:
    - Pass in logger/output sinks; keep core free of side effects.
  - Functional style:
    - Core computes values; boundary decides how to handle them (print, persist, send, etc.).
  */

  // 13) Property‑Based Testing (outline): Functor identity/composition
//  import org.scalacheck.Prop.forAll
//
//  object PropertyTests {
//    // Functor identity law
//    val functorIdentity = forAll { (xs: List[Int]) =>
//      xs.map(identity) == xs
//    }
//
//    // Functor composition law
//    val functorComposition = forAll { (xs: List[Int], f: Int => Int, g: Int => Int) =>
//      xs.map(f compose g) == xs.map(g).map(f)
//    }
//  }
  /*
  1. Lines: properties, not examples — test many random inputs

  - Code:
    - functorIdentity: forAll { (xs: List[Int]) => xs.map(identity) == xs }
    - functorComposition: forAll { (xs: List[Int], f, g) => xs.map(f compose g) == xs.map(g).map(f) }
  - Meaning:
    - Property-based testing: ScalaCheck generates many random inputs.
    - Each property must hold for all generated cases, not just hand‑picked examples.

  2. The properties being checked

  - Functor identity law:
    - xs.map(identity) == xs
    - Mapping with identity must not alter the list.
  - Functor composition law:
    - xs.map(f compose g) == xs.map(g).map(f)
    - Mapping with composed functions equals successive mapping.

  3. Why property‑based tests are powerful

  - Broad input coverage:
    - Random lists include edge cases (empty, large, duplicates, negatives).
  - Law‑driven:
    - Express algebraic law, let framework validate it automatically.
  - Shrinking:
    - On failure, input is minimized to smallest failing counterexample.

  4. Purity and determinism

  - Pure functions (map, identity, composition) are deterministic.
  - Failures reproducible given the same random seed.

  5. Extending the idea

  - Monoid laws with List concatenation (identity, associativity).
  - Option laws: map(identity) == self, map(f).map(g) == map(g compose f).
  - Monad laws with flatMap and pure.

  6. Edge cases

  - Null values: rarely in Scala unless via Java interop.
  - Large lists: may require size limits.
  - Equality: Scala’s == checks structural equality (order + contents).

  7. Minimal runnable shape

  - val functorIdentity = forAll { xs: List[Int] => xs.map(identity) == xs }
  - val functorComposition = forAll { (xs: List[Int], f: Int => Int, g: Int => Int) =>
      xs.map(f compose g) == xs.map(g).map(f)
    }

  8. Failure meaning

  - Identity failure: identity mapping changed the list (bug).
  - Composition failure: composition law not preserved, possible issue with function definitions or transformations.
  */

}