"""
 FP CONCEPTS (Python 3.10+) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — lambdas, currying via closures, partials.
 2) Composition (∘) — compose(f, g)(x) == f(g(x)).
 3) Referential Transparency — pure expressions can be replaced by their values.
 4) Immutability — frozen dataclasses / copy functions; avoid in‑place mutation.
 5) Higher‑Order Functions — map/filter/reduce pipelines.
 6) Functor — map on custom Maybe preserving structure.
 7) Applicative — ap / liftA2 on Option to combine independent contexts.
 8) Monad — flat_map for dependent sequencing.
 9) Natural Transformation — list -> Maybe (head).
 10) Monoid — reduce with associative op and identity.
 11) ADTs & Pattern Matching — dataclasses + match (3.10+).
 12) Effects at the Edges — keep core pure; perform I/O in main.
 13) Property‑Based Testing — outline with hypothesis.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from functools import reduce
from typing import Callable, Generic, Iterable, Optional as Opt, TypeVar, List, Tuple


A = TypeVar('A'); B = TypeVar('B'); C = TypeVar('C')

# 1) Lambda, Application, Currying, Partial Application
inc: Callable[[int], int] = lambda x: x + 1
add: Callable[[int], Callable[[int], int]] = lambda x: (lambda y: x + y)
add5 = add(5)                # partial
seven = add5(2)              # 7
"""
1. Line 1: a simple lambda (a function value)

- Code: inc: Callable[[int], int] = lambda x: x + 1
- Meaning:
- inc is a function that takes an int and returns an int.
- The lambda x: x + 1 is an anonymous function that increments its argument.

- Usage:
- inc(10) == 11

- Notes:
- The Callable[[int], int] part is just a type hint; Python doesn’t enforce it at runtime.
- The function is pure: same input → same output, no side effects.

2. Line 2: a curried function (a function that returns a function)

- Code: add: Callable[[int], Callable[[int], int]] = lambda x: (lambda y: x + y)
- Meaning:
- add takes one argument x and returns another function that takes y and computes x + y.
- This is currying via a closure: the inner lambda “remembers” x from the outer scope.

                                                   - Why this is useful:
- Enables partial application—fix x now, supply y later.
- Composes nicely with other higher‑order functions.

3. Line 3: partial application (fix the first argument)

- Code: add5 = add(5)
- Meaning:
- Call add with x = 5, producing a new function y -> 5 + y.
- add5 now has type Callable[[int], int] (by intent; Python is dynamic).

- Usage:
- add5(10) == 15

4. Line 4: function application (call the specialized function)

- Code: seven = add5(2)
- Meaning:
- Calls the function returned earlier with y = 2 → computes 5 + 2 = 7.
- seven == 7

Equivalences and tips
- Direct two‑step application:
- add(5)(2) == 7

- Uncurried vs curried:
- Uncurried version would look like add2 = lambda x, y: x + y (takes both args at once).

- Closures in action:
- The inner lambda y: x + y captures x from the outer lambda’s environment, which is why add(5) “remembers” 5.

- Type hints:
- These annotations help tools (linters, type checkers) but don’t change runtime behavior. If you’re using them, import Callable from typing:
- from typing import Callable
"""

# 2) Composition (∘)
def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    return lambda a: f(g(a))

double = lambda x: x * 2
comp_res = compose(inc, double)(10)  # 21
"""
1. Line 1: define a generic compose

- Code: def compose(f, g) -> ...: return lambda a: f(g(a))
- Meaning:
- Given g: A -> B and f: B -> C, compose builds h: A -> C such that h(a) = f(g(a)).
- It returns a new function that first applies g, then applies f to the result.

- Why useful:
- Lets you build pipelines without intermediate variables (“do g first, then f”).

- Note on types:
- Type hints are for readability/tools; Python enforces them only if you run a type checker.

2. Line 2: define a “double” function

- Code: double = lambda x: x * 2
- Meaning:
- A simple function number -> number that multiplies input by 2.

- Usage:
- double(7) == 14

3. Line 3: apply the composed function

- Code: comp_res = compose(inc, double)(10)
- Step-by-step trace:
1. compose(inc, double) returns a new function h(a) = inc(double(a)).
2. h(10) calls double(10) → 20.
3. Then calls inc(20) → 21.
4. comp_res == 21.

- Assumption:
- inc is defined (e.g., inc = lambda x: x + 1).

Extra notes
- Order matters:
- compose(f, g)(x) = f(g(x)) ⇒ “run g first, then f.”

- Duck typing:
- As long as double’s output is valid input for inc, the composition works (no explicit generics required at runtime).
"""

# 3) Referential Transparency
pure_total = sum([1,2,3])            # == 6; safe replacement
# print("hi")  # side effect → not referentially transparent
"""
1. What “referential transparency” means

- An expression is referentially transparent if you can replace it with its value everywhere in the program without changing the program’s behavior.
- In practice: no side effects, no hidden/external state, deterministic (same input → same output).

2. The pure expression

- Code: sum([1,2,3])
- Meaning:
- Builds a list [1, 2, 3].
- sum folds the elements with addition starting from 0.

- Micro-trace:
- Start: 0
- 0 + 1 = 1
- 1 + 2 = 3
- 3 + 3 = 6
- Result: 6

- Why it’s referentially transparent:
- No I/O, no mutation, no randomness.
- You can safely replace sum([1,2,3]) with 6 anywhere. The program’s behavior won’t change.

- Therefore: pure_total is 6, and writing pure_total = 6 is behaviorally equivalent.

3. The side‑effecting expression (not referentially transparent)

- Code: print("hi")
- Why it’s not referentially transparent:
- It performs I/O by printing to stdout.
- Replacing it with its “value” (it returns None) would remove the print, changing observable behavior.
- So it cannot be freely substituted by a value without altering the program.

Key takeaways
- Pure expressions are interchangeable with their values—great for reasoning, testing, and refactoring.
- Side effects (like printing) are not referentially transparent; they change the outside world and cannot be replaced by a value without changing behavior.
"""

# 4) Immutability via frozen dataclasses
@dataclass(frozen=True)
class Point:
    x: int
    y: int

p1 = Point(1,1)
p2 = replace(p1, x=2)                # new instance; p1 unchanged
"""
1. Line 1–4: define an immutable data carrier (a frozen dataclass)

- Code: @dataclass(frozen=True) class Point: x: int; y: int
- What a frozen dataclass is:
- A compact class for “data with value semantics”.
- Automatically generates:
- An **init**(x: int, y: int)
- Equality (**eq**) and a readable **repr**
- A hash (**hash**) when frozen=True and fields are hashable

- frozen=True enforces read‑only fields; attempts to assign (e.g., p.x = ...) raise dataclasses.FrozenInstanceError.

- Immutability here:
- You can’t do p.x = ... or p.y = ...; instances are immutable.
- Note: immutability is shallow—if a field holds a mutable object, the dataclass won’t freeze that object. Using ints here is fully safe.

2. Line 5: create the first instance

- Code: p1 = Point(1,1)
- Meaning:
- Calls the generated constructor with x = 1, y = 1.
- p1 is a reference to an immutable Point(1, 1).
- Attribute access: p1.x == 1, p1.y == 1.

3. Line 6: create a new instance instead of mutating

- Code: p2 = replace(p1, x=2)
- Step‑by‑step:
1. Read p1.y (implicitly retained by replace since we don’t override y) → 1 (read, not mutate).
2. replace constructs a new Point with x overridden to 2 and y copied from p1 → Point(2, 1).
3. Bind it to p2.

- Key idea:
- You didn’t “change” p1; you built a new value (p2) that shares y with p1 but has a different x.
- p1 remains Point(1, 1). p2 is Point(2, 1).

Why this matters in FP
- Immutability avoids accidental shared‑state bugs.
- Values are safe to pass around and reuse; reasoning is simpler (no hidden changes).
- Works well with parallelism and caching.
    - If you need a “modified” version, you create a new instance with the desired changes (as shown with p2).
"""

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')

# 5) Higher‑Order Functions (map/filter/reduce)
sum_filtered = reduce(
    lambda a, b: a + b,
    filter(lambda x: x > 2, map(lambda x: x * 2, [1, 2, 3])),
    0
)
"""
1. Built-in HOF pipeline (map → filter → reduce)
- What happens:
  - Start with [1, 2, 3]
  - map(lambda x: x*2)       → [2, 4, 6]
  - filter(lambda x: x > 2)  → [4, 6]
  - reduce(lambda a,b: a+b, ..., 0) → 0 + 4 + 6 = 10

- Why these are HOFs:
  - map, filter, reduce all receive functions (lambdas) as parameters, so they’re higher‑order functions.

- Complexity:
  - One pass per stage; overall linear in input size.

- Purity note:
  - If the lambdas are pure, the whole pipeline is pure and deterministic.
"""

# Custom HOF #1: map for list (re-implementing map to show the idea)
def map_list(xs: Iterable[A], f: Callable[[A], B]) -> List[B]:
    return [f(a) for a in xs]

# Usage: doubles each element -> [2, 4, 6]
custom_doubled: List[int] = map_list([1, 2, 3], lambda x: x * 2)
"""
2. Custom HOF #1: map_list

- Purpose:
  - Transform each element independently using a provided function, producing a new list of possibly different element type.

- Signature reasoning:
  - map_list(xs: Iterable[A], f: A -> B) -> List[B]
  - A is input element type; B is output element type; f captures the behavior.

- Behavior (step-by-step):
  - Iterate xs; for each a, compute f(a) and collect into a new list.

- Example:
  - map_list([1,2,3], lambda x: x*2) → [2, 4, 6]
  - map_list(["a","bb"], len) → [1, 2]

- Complexity:
  - Time O(n), Space O(n), where n = len(xs).

- Why it’s higher‑order:
  - It takes a function f as an argument.

- Purity advice:
  - Keep f pure for predictable, testable behavior.
"""

# Custom HOF #2: map -> filter -> reduce as one reusable helper
def map_filter_reduce(
        xs: Iterable[A],
        mapper: Callable[[A], A],
        predicate: Callable[[A], bool],
        zero: B,
        reducer: Callable[[B, A], B],
) -> B:
    acc: B = zero
    for a in xs:
        m = mapper(a)
        if predicate(m):
            acc = reducer(acc, m)
    return acc

# Usage: same logic as the pipeline above -> 10
custom_mfr: int = map_filter_reduce(
    [1, 2, 3],
    mapper=lambda x: x * 2,     # [2, 4, 6]
    predicate=lambda x: x > 2,  # [4, 6]
    zero=0,
    reducer=lambda a, b: a + b  # 10
)
"""
3. Custom HOF #2: map_filter_reduce (one reusable helper)

- Purpose:
  - Perform a “map → filter → reduce” pipeline in a single pass to avoid intermediate structures.

- Signature reasoning:
  - mapper: A -> A (keeps one element type flowing through; can be generalized if needed)
  - predicate: A -> bool (selects which mapped values to keep)
  - zero: initial accumulator value of type B
  - reducer: (B, A) -> B (folds kept elements into the accumulator)

- Behavior (step-by-step):
  - acc = zero
  - For each a in xs:
    - m = mapper(a)
    - If predicate(m): acc = reducer(acc, m)
  - Return acc

- Example (matches the built‑in pipeline):
  - xs = [1,2,3]
  - mapper = x*2 → [2,4,6]
  - predicate = x>2 → keep [4,6]
  - zero = 0
  - reducer = a+b → 10

- Complexity:
  - Single traversal: Time O(n), Space O(1) extra.

- Why it’s higher‑order:
  - Accepts three behaviors: mapper, predicate, reducer.

- Purity advice:
  - Keep the three functions pure to preserve determinism and ease testing.

- Variations:
  - Generalize mapper to A -> C, adjust predicate/reducer accordingly.
  - Add early termination by making reducer return a short‑circuiting wrapper (e.g., sentinel object).
"""

# Custom HOF #3: return a new function by repeating an operation n times
def repeat(n: int, f: Callable[[T], T]) -> Callable[[T], T]:
    def composed(t: T) -> T:
        r = t
        for _ in range(n):
            r = f(r)
        return r
    return composed

# Usage: reuse an `inc` function, apply it 3 times
inc: Callable[[int], int] = lambda x: x + 1
inc3 = repeat(3, inc)
inc3_res: int = inc3(10)  # 13
"""
4. Custom HOF #3: repeat (returns a new function)

- Purpose:
  - Build a new function that applies f to its input n times:
    repeat(n, f)(x) = f(f(...f(x)...)) with n applications.

- Signature reasoning:
  - f must be an endofunction on T (T -> T) so it composes with itself.

- Behavior (step-by-step):
  - Return a closure composed(t):
    - r = t
    - For _ in range(n): r = f(r)
    - Return r

- Examples:
  - inc = lambda x: x + 1
    repeat(3, inc)(10) → 13
  - exclaim = lambda s: s + "!"
    repeat(3, exclaim)("go") → "go!!!"

- Complexity:
  - Each call to the returned function does n applications of f (O(n)).

- Why it’s higher‑order:
  - Takes a function and returns a new function.

- Edge cases:
  - n = 0 → identity function (returns input unchanged).

Key takeaways
- All these are higher-order because they take functions as parameters or return functions.
- The built-in example uses standard HOFs; the custom examples show how to build your own reusable functional utilities.
- map_filter_reduce packages a common pipeline into a single, composable abstraction.
- repeat builds new behavior (n-fold application) by returning a composed function.
"""

# 6) Functor: minimal Maybe with map
class Maybe(Generic[A]):
    def __init__(self, value: Opt[A]):
        self.value = value
    def map(self, f: Callable[[A], B]) -> 'Maybe[B]':
        """Apply f inside the context, preserving structure."""
        return Maybe(None if self.value is None else f(self.value))
    def __repr__(self) -> str:
        return f"Just({self.value})" if self.value is not None else "Nothing"
"""
-------------------------------------------------------------------------------
Explanation of Functor behavior in Python's Maybe
-------------------------------------------------------------------------------
1. Code: Maybe(42).map(lambda x: x + 1)
   - Constructs a Maybe containing 42 (Just 42).
   - map applies the lambda (x+1) to the inner value if present.
   - Result: Just(43).

2. Functor definition (informal):
   A type constructor F<_> that supports a structure-preserving mapping operation:
     map: (A -> B) -> F<A> -> F<B>
   - For Maybe:
     F = Maybe
     map(f, Maybe[A]) -> Maybe[B]
     "Preserve structure" means: we keep the container (Just/Nothing) as-is.

3. Why useful:
   - Avoids manual None checks.
     Instead of:
       if x is not None: return Maybe(f(x)) else: return Maybe(None)
     You can simply write:
       maybe.map(f)
   - Encourages safe, composable transformations.

4. Behavior summary:
   - Present: Maybe(v).map(f) == Maybe(f(v))
   - Empty:   Maybe(None).map(f) == Maybe(None)

5. Functor laws:
   - Identity:       maybe.map(lambda x: x) == maybe
   - Composition:    maybe.map(lambda x: g(f(x))) == maybe.map(f).map(g)
     (for pure functions f and g)

6. Purity:
   - If f is pure, then map is pure and referentially transparent.
   - Same input + same function => same output.

7. Edge cases:
   - Maybe(None) acts like "Nothing": map does nothing, stays Nothing.
   - Costs: O(1) (single function call + wrapper).
   - map vs flat_map:
     - map:    f: A -> B  => Maybe[B]
     - flatMap: f: A -> Maybe[B] to avoid nesting Maybe[Maybe[B]].

8. Chaining example:
   Maybe(10)
     .map(lambda x: x + 1)   # Just(11)
     .map(lambda x: x * 2)   # Just(22)
   Maybe(None)
     .map(lambda x: x + 1)   # Nothing
-------------------------------------------------------------------------------
"""

# 7) Applicative: Option with ap and liftA2
class Option(Maybe[A]):
    def ap(self: 'Option[Callable[[A], B]]', oa: 'Option[A]') -> 'Option[B]':
        if self.value is None or oa.value is None:
            return Option(None)
        return Option(self.value(oa.value))

def pure(x: A) -> Option[A]:
    return Option(x)

# Ensure binary functions are curried so ap can apply one argument at a time
def curry2(f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    return lambda a: (lambda b: f(a, b))

def liftA2(f: Callable[[A, B], C]) -> Callable[[Option[A]], Callable[[Option[B]], Option[C]]]:
    # Curry f so that each ap applies exactly one argument
    return lambda oa: lambda ob: pure(curry2(f)).ap(oa).ap(ob)

name: Option[str] = Option("Ada")
age:  Option[int] = Option(36)
mk_user = lambda n, a: {"name": n, "age": a}
user_opt: Option[dict] = liftA2(mk_user)(name)(age)      # Just({'name': 'Ada', 'age': 36})
"""
-------------------------------------------------------------------------------
Explanation — Applicative Option with `ap`, `curry2`, and `liftA2`
-------------------------------------------------------------------------------
1) Lines and meaning
   - `ap`: apply a function **inside** Option to a value **inside** Option.
     • Type shape: Option[(A -> B)] -> Option[A] -> Option[B].
     • If either side is None → result is None; if both are Some → apply function.
   - `pure(x)`: lift a plain value into the Option context (Some x).
   - `curry2(f)`: convert a binary function f(a,b) into chained single-arg calls: a -> (b -> f(a,b)).
     • Needed so that `ap` (which applies one argument) can apply f in two steps.
   - `liftA2(f)`: lift a binary function `(A, B) -> C` to operate on Option[A], Option[B].
     • Implementation: `pure(curry2(f)).ap(oa).ap(ob)`.
     • Reads as: put curried f in Option, apply to oa, then apply to ob.

2) Why “Applicative” (intuition)
   - Applicative combines **independent** contexts without sequencing dependencies.
   - With Option, the context is presence/absence; combination succeeds only if all are present.
   - Signature shape mirrors classical `map2`: (A,B)->C, F[A], F[B] => F[C].

3) How it works step-by-step (for `user_opt`)
   - `curry2(mk_user)` yields `a -> (b -> mk_user(a,b))`.
   - `pure(curry2(mk_user))` puts that curried function into Option: Some(curried).
   - First `ap(oa)` where `oa = name = Some("Ada")` → Some(b -> mk_user("Ada", b)).
   - Second `ap(ob)` where `ob = age = Some(36)` → Some(mk_user("Ada", 36)).
   - Result: `Some({"name":"Ada", "age":36})`.

4) Behavior summary
   - Present-present: `Some(f).ap(Some(a)).ap(Some(b)) = Some(f(a)(b))`.
   - Any None: result is None; function is not executed for missing arg(s).

5) Why not just `map`?
   - `map` handles `(A -> B)` with one input. For two inputs you need either:
     • `map2` (provided by libraries), or
     • `ap` + curried functions (`liftA2`) as shown here.

6) Purity & referential transparency
   - If `f` is pure, the whole pipeline is pure: same inputs → same outputs.
   - Side effects in `f` (if any) run only when all inputs are present.

7) Edge cases & tips
   - Use `Option(x)`/Maybe(None) style to avoid None vs Some(None) confusion.
   - This pattern generalizes to `liftA3`, `liftA4`, etc., by currying higher‑arity functions.

8) Small variations
   - Pointfree style: `liftA2(lambda n: lambda a: mk_user(n,a))(name)(age)` (already achieved via curry2).
   - Building tuples: `liftA2(lambda a,b: (a,b))(oa)(ob)` → Some((a,b)).

9) Applicative laws (intuition)
   - Identity:      pure(id).ap(v) == v
   - Homomorphism:  pure(f).ap(pure(x)) == pure(f(x))
   - Interchange:   u.ap(pure(y)) == pure(f => f(y)).ap(u)
   - Composition:   pure(comp).ap(u).ap(v).ap(w) == u.ap(v.ap(w))
     (These hold assuming a lawful implementation; shown here in spirit.)
-------------------------------------------------------------------------------
"""

# 8) Monad: flat_map for dependent sequencing
class OptionM(Option[A]):
    def flat_map(self, f: Callable[[A], 'OptionM[B]']) -> 'OptionM[B]':
        return self if self.value is None else f(self.value)

monad_res: OptionM[int] = OptionM(2).flat_map(lambda x: OptionM(3).map(lambda y: x + y))  # Just(5)
"""
1. Line: flat_map sequences dependent computations (Monad behavior)

- Code: OptionM(2).flat_map(lambda x: OptionM(3).map(lambda y: x + y))
- Meaning:
  - Start with OptionM(2).
  - flat_map "unboxes" x if present, then runs the next computation which may also produce OptionM.
  - Inside, map transforms OptionM(3) by adding x, yielding OptionM(x + 3).
  - If the outer OptionM were empty, the lambda wouldn’t run and the result would be OptionM(None).

- Result here:
  - x = 2; y = 3 → x + y = 5 → OptionM(5).

2. What "Monad" means (informal)
- A type constructor M<_> with flat_map/bind to chain computations returning M:
  - flat_map: (A -> M<B>) -> M<A> -> M<B>
- For OptionM:
  - M = OptionM
  - flat_map allows the next step to depend on the value from the previous step and may short‑circuit on None.

3. Why flat_map (vs map) here
- map: A -> B (wraps B back into OptionM automatically).
- flat_map: A -> OptionM<B> (you already return an OptionM; flat_map prevents nesting OptionM[OptionM[B]]).
- Pattern:
  - optA.flat_map(lambda a: optB.map(lambda b: combine(a,b)))

4. Behavior summary
- Present case:
  - OptionM(a).flat_map(f) == f(a)
- Empty case:
  - OptionM(None).flat_map(f) == OptionM(None)
- Short‑circuiting:
  - If any step is None, the rest of the chain is skipped.

5. Sequencing dependent steps (typical shapes)
- Two steps:
  - oa.flat_map(lambda a: ob.map(lambda b: f(a,b)))
- Three steps:
  - oa.flat_map(lambda a: ob.flat_map(lambda b: oc.map(lambda c: g(a,b,c))))

6. Purity and referential transparency
- If the functions passed to flat_map/map are pure:
  - Same inputs → same outputs; no side effects.
- Side effects run only when prior OptionMs are present.

7. Edge cases and tips
- None values:
  - Avoid wrapping raw None unless intended; use OptionM(None) carefully.
- Readability:
  - For multiple steps, prefer intermediate functions or comprehensions if available.

8. Monad laws (for predictable refactoring)
- Left identity:  OptionM(a).flat_map(f) == f(a)
- Right identity: opt.flat_map(OptionM) == opt
- Associativity:  opt.flat_map(f).flat_map(g) == opt.flat_map(lambda a: f(a).flat_map(g))
  - These hold with pure functions and ensure predictable chaining.
"""

# 9) Natural Transformation: list -> Maybe (head)
def head_maybe(xs: list[A]) -> Maybe[A]:
    return Maybe(xs[0]) if xs else Maybe(None)
'''
1. Line: a natural transformation between containers (list -> Maybe)

- Code: head_maybe(xs)
- Meaning:
  - Converts a list[A] into a Maybe[A] by taking its first element if present.
  - If the list is empty, returns Maybe(None).
  - Otherwise, wraps the first element in Maybe(...).

- Result examples:
  - head_maybe([10, 20])  → Maybe(10)
  - head_maybe([])        → Maybe(None)

2. What “Natural Transformation” means (informal)
- A uniform, structure-preserving mapping between type constructors F and G:
  - For all A, a function nat: F[A] -> G[A] that does not depend on the specific type A.
- Here:
  - F = list, G = Maybe, nat = head_maybe
  - Uniform: works the same for ints, strings, or custom objects.

3. Why this is useful
- Changes context, not the inner type:
  - Transforms “many (possibly zero)” (list) into “zero or one” (Maybe).
- Useful when APIs expect Maybe/Optional instead of raw lists.

4. Behavior and properties
- Total function for any list[A]:
  - [] → Maybe(None)
  - [x, ...] → Maybe(x)
- Pure and referentially transparent:
  - Same input → same output, no side effects.

5. Edge cases and tips
- Empty list always yields Maybe(None).
- If xs[0] could be None itself, the result is Maybe(None), consistent with empty.
- Performance:
  - O(1) time/O(1) space; only checks emptiness and maybe reads index 0.

6. Variations
- last_maybe:
  - Return the last element instead of the first.
- safeHead for tuples/arrays:
  - len(xs) == 0 ? Maybe(None) : Maybe(xs[0])

7. Composition intuition
- Natural transformations compose:
  - If you had another transformation Maybe[A] -> Either[E, A], composing it with head_maybe yields list[A] -> Either[E, A], still uniform and type-parametric.
'''

# 10) Monoid: associative op + identity
m_sum = reduce(lambda a,b: a+b, [1,2,3], 0)
m_str = reduce(lambda a,b: a+b, ["a","b","c"], "")
'''
1. Line: reduce with a Monoid (operation + identity)

- Code:
  - reduce(lambda a,b: a+b, [1,2,3], 0)
  - reduce(lambda a,b: a+b, ["a","b","c"], "")
- Meaning:
  - A Monoid is a pair (⊕, e) with:
    - Associative binary operation ⊕: (x ⊕ y) ⊕ z == x ⊕ (y ⊕ z)
    - Identity element e: e ⊕ x == x == x ⊕ e
  - For ints under addition: (⊕ = +, e = 0)
  - For strings under concatenation: (⊕ = +, e = "")

- Why reduce needs a Monoid:
  - Folding a collection safely requires an operation that can combine elements in any grouping (associativity) and a neutral start value (identity).

2. Behavior (step-by-step)

- m_sum:
  - Start acc = 0
  - 0 + 1 = 1
  - 1 + 2 = 3
  - 3 + 3 = 6
  - Result: 6

- m_str:
  - Start acc = ""
  - "" + "a" = "a"
  - "a" + "b" = "ab"
  - "ab" + "c" = "abc"
  - Result: "abc"

3. Why associativity and identity matter

- Associativity:
  - Reduce may regroup operations internally; associativity guarantees the same result.
  - Example: (("a" + "b") + "c") == ("a" + ("b" + "c")) == "abc"

- Identity:
  - The seed value must be neutral so it doesn’t affect the outcome:
    - 0 + x = x
    - "" + s = s

4. Purity and determinism

- With pure operations (+ on ints/strings), reduce is pure and referentially transparent:
  - Same input → same output; no side effects.

5. Edge cases and tips

- Empty input:
  - reduce(op, [], identity) returns the identity (0 or "")
- Non-associative ops:
  - Avoid using non-associative operations (e.g., subtraction, floating‑point addition with rounding issues) for reduce.
- Performance:
  - String concatenation in Python is efficient for small lists, but for large concatenations prefer `"".join(list)`.

6. Custom Monoids (examples)

- Product monoid (ints):
  - op = (a, b) -> a * b, identity = 1
- Max/Min monoids:
  - op = max with identity = float('-inf') (for max)
  - op = min with identity = float('inf') (for min)

7. General intuition

- Monoid abstraction ensures safe, consistent reductions over collections.
- Python’s reduce + identity fits perfectly into this pattern.
'''

# 11) ADTs & Pattern Matching
@dataclass(frozen=True)
class Circle: r: float
@dataclass(frozen=True)
class Rect:   w: float; h: float
Shape = Circle | Rect

def area(s: Shape) -> float:
    match s:
        case Circle(r):
            return 3.14159 * r * r
        case Rect(w, h):
            return w * h
"""
1. Lines 4–12: define an Algebraic Data Type (ADT) using dataclasses and a union type

- Code:
  - @dataclass(frozen=True) class Circle: r: float
  - @dataclass(frozen=True) class Rect: w: float; h: float
  - Shape = Circle | Rect
- Meaning:
  - Dataclasses with frozen=True create immutable data carriers with auto‑generated constructor, equality, and repr.
  - Circle and Rect are the only variants, so Shape is a closed set: Shape = Circle | Rect.
  - The union type (|) models a sum type: a Shape must be one of these two cases.

- Why immutability matters:
  - Prevents mutation, simplifying reasoning and making values thread‑safe.
  - Ensures shapes remain consistent once created.

2. Lines 14–19: structural pattern matching with match

- Code:
  - match s: case Circle(r): ...; case Rect(w,h): ...
- Meaning:
  - Deconstructs s according to its type and binds its fields.
  - Python 3.10+ supports this pattern matching, similar to switch in other languages.
  - Unlike Java/Scala, Python doesn’t enforce exhaustiveness at compile time, but explicit matches provide clarity.

3. Behavior (step‑by‑step)

- If s is Circle(r):
  - area = π * r * r
- If s is Rect(w,h):
  - area = w * h

4. Why ADTs + pattern matching are powerful

- Clarity:
  - Domain is expressed as a finite set of variants.
- Safety:
  - All known shapes are handled explicitly in one place.
- Immutability:
  - Frozen dataclasses guarantee stable, read‑only data.
- Refactoring:
  - Adding a new Shape requires updating Shape’s union and all matches—tests/static tools reveal missing cases.

5. Purity and reasoning

- area is pure:
  - No side effects; output depends only on input.
- Referential transparency:
  - Example: area(Circle(2)) = 12.56636, which can safely replace the function call.

6. Edge cases and tips

- Negative/NaN values:
  - Not validated by default—add checks if desired.
- Extensibility:
  - To add Triangle, define a Triangle dataclass and extend Shape accordingly.
- Performance:
  - Pattern matching is O(1) dispatch. Dataclasses are efficient and lightweight.
"""

# 12) Effects at the Edges — pure core + I/O boundary
class Domain:
    @staticmethod
    def pure_logic(x: int) -> int:
        return x * 2

def main_io() -> None:
    print(Domain.pure_logic(5))  # side effect at boundary

if __name__ == "__main__":
    main_io()
"""
1. Lines: separate pure core from I/O

- Code:
  - class Domain: defines pure_logic (deterministic, no I/O, no mutation).
  - main_io: performs I/O (printing result) at the boundary.
  - __main__ guard: ensures effect only runs when script executed directly.
- Meaning:
  - Keep computation (pure_logic) side‑effect free.
  - Only perform I/O (print) at program boundary.

2. Why this separation matters

- Testability:
  - pure_logic is easy to unit test: same input → same output.
- Reasoning & refactoring:
  - Pure code is referentially transparent, easier to refactor.
- Reuse & composition:
  - Pure functions compose well; you can reuse them in pipelines.
- Reliability & observability:
  - Side effects are centralized in main_io, easy to monitor.

3. Behavior (step‑by‑step)

- Domain.pure_logic(5) → 10 (pure computation).
- main_io calls print(10) → writes "10" to console (effect).

4. Purity vs effects

- pure_logic:
  - No global state, no I/O, deterministic; returns x * 2.
- print:
  - Performs I/O; not referentially transparent.

5. Edge cases and tips

- Exceptions:
  - Handle around I/O (in main_io); keep core pure.
- Time, randomness, config:
  - Pass them as parameters, not inside pure functions.
- None/null:
  - Use Optional typing or validation at the edges.

6. Testing strategy

- Unit tests:
  - assert Domain.pure_logic(5) == 10
- Integration tests:
  - Capture stdout and check output of main_io.

7. Refactoring patterns

- Dependency injection at edges:
  - Pass a writer/log sink into main_io.
- Functional style:
  - Core computes; boundary decides what to do (print, save, send).
"""

# 13) Property‑Based Testing (outline with hypothesis)
# from hypothesis import given, strategies as st
# @given(st.lists(st.integers()))
# def test_functor_identity(xs):
#     assert list(map(lambda x: x, xs)) == xs
'''
1. Line: a property, not an example — test many random inputs

- Code:
  - @given(st.lists(st.integers()))
  - def test_functor_identity(xs): ...
- Meaning:
  - Property-based test: Hypothesis generates many random lists of integers.
  - The assertion must hold for all generated inputs, not just a few hand‑picked cases.

2. The property being checked (Functor identity law for map)

- Law:
  - xs.map(identity) == xs
- In Python:
  - list(map(lambda x: x, xs)) should equal xs
- Intuition:
  - Mapping with the identity function should not change the structure or the values.

3. Why property‑based tests are powerful

- Broad input coverage:
  - Randomized data explores edge cases (empty list, large lists, negatives, duplicates).
- Less bias:
  - You specify the law; Hypothesis probes many cases automatically.
- Shrinking:
  - On failure, Hypothesis minimizes input to a smallest counterexample, easing debugging.

4. Determinism and purity

- Hypothesis tests are deterministic given a seed.
- Pure transformations like identity/map are ideal for property checks.

5. Extending the idea (other fundamental laws)

- Functor composition:
  - list(map(lambda x: f(g(x)), xs)) == list(map(f, map(g, xs)))
- Monoid laws (with reduce):
  - Identity: reduce(op, [], e) == e
  - Associativity: grouping does not change result
- Option laws:
  - map(identity) == self
  - map(f).map(g) == map(lambda x: g(f(x)))

6. Edge cases

- Nulls: Python lists can contain None; decide whether to allow.
- Large lists: performance considerations; Hypothesis limits sizes.
- Equality: use == for deep equality of lists.

7. Minimal runnable shape

- @given(st.lists(st.integers()))
- def test_functor_identity(xs):
-   assert list(map(lambda x: x, xs)) == xs

8. What a failure would mean

- If this property fails:
  - Either map(identity) changed the list (unexpected), or comparison isn’t as expected.
  - Hypothesis will report failing input and minimized counterexample.
'''
