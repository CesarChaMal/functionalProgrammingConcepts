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
from typing import Callable, Generic, Optional as Opt, TypeVar

A = TypeVar('A'); B = TypeVar('B'); C = TypeVar('C')

# 1) Lambda, Application, Currying, Partial Application
inc: Callable[[int], int] = lambda x: x + 1
add: Callable[[int], Callable[[int], int]] = lambda x: (lambda y: x + y)
add5 = add(5)                # partial
seven = add5(2)              # 7

# 2) Composition (∘)
def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    return lambda a: f(g(a))

double = lambda x: x * 2
comp_res = compose(inc, double)(10)  # 21

# 3) Referential Transparency
pure_total = sum([1,2,3])            # == 6; safe replacement
# print("hi")  # side effect → not referentially transparent

# 4) Immutability via frozen dataclasses
@dataclass(frozen=True)
class Point:
    x: int
    y: int

p1 = Point(1,1)
p2 = replace(p1, x=2)                # new instance; p1 unchanged

# 5) Higher‑Order Functions (map/filter/reduce)
sum_filtered = reduce(lambda a,b: a+b, filter(lambda x: x>2, map(lambda x: x*2, [1,2,3])), 0)

# 6) Functor: minimal Maybe with map
class Maybe(Generic[A]):
    def __init__(self, value: Opt[A]):
        self.value = value
    def map(self, f: Callable[[A], B]) -> 'Maybe[B]':
        """Apply f inside the context, preserving structure."""
        return Maybe(None if self.value is None else f(self.value))
    def __repr__(self) -> str:
        return f"Just({self.value})" if self.value is not None else "Nothing"

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

# 8) Monad: flat_map for dependent sequencing
class OptionM(Option[A]):
    def flat_map(self, f: Callable[[A], 'OptionM[B]']) -> 'OptionM[B]':
        return self if self.value is None else f(self.value)

monad_res: OptionM[int] = OptionM(2).flat_map(lambda x: OptionM(3).map(lambda y: x + y))  # Just(5)

# 9) Natural Transformation: list -> Maybe (head)
def head_maybe(xs: list[A]) -> Maybe[A]:
    return Maybe(xs[0]) if xs else Maybe(None)

# 10) Monoid: associative op + identity
m_sum = reduce(lambda a,b: a+b, [1,2,3], 0)
m_str = reduce(lambda a,b: a+b, ["a","b","c"], "")

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

# 12) Effects at the Edges — pure core + I/O boundary
class Domain:
    @staticmethod
    def pure_logic(x: int) -> int:
        return x * 2

def main_io() -> None:
    print(Domain.pure_logic(5))  # side effect at boundary

if __name__ == "__main__":
    main_io()

# 13) Property‑Based Testing (outline with hypothesis)
# from hypothesis import given, strategies as st
# @given(st.lists(st.integers()))
# def test_functor_identity(xs):
#     assert list(map(lambda x: x, xs)) == xs