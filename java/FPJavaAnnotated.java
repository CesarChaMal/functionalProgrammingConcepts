/*
 FP CONCEPTS (Java 17+) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — Lambdas + Function composition.
 2) Composition (∘) — Use Function#compose/andThen.
 3) Referential Transparency — Pure expressions can be substituted by values.
 4) Immutability — Records are concise immutable carriers.
 5) Higher‑Order Functions — Stream pipelines with map/filter/reduce.
 6) Functor — Optional.map.
 7) Applicative — map2 to combine independent Optionals.
 8) Monad — Optional.flatMap for dependent sequencing.
 9) Natural Transformation — List<A> -> Optional<A>.
 10) Monoid — Reduce with associative op and identity.
 11) ADTs & Pattern Matching — sealed interfaces + switch.
 12) Effects at the Edges — Keep core pure; do I/O in main.
 13) Property‑Based Testing — Outline (jqwik).
*/

import java.util.*;
import java.util.function.*;

public class FPJavaAnnotated {
  // 1) Lambda, Application, Currying, Partial Application
  static final Function<Integer,Integer> inc = x -> x + 1;                      // λx. x+1
  static final Function<Integer,Function<Integer,Integer>> add = x -> y -> x + y; // curried
  static final Function<Integer,Integer> add5 = add.apply(5);                   // partial
  static final int seven = add5.apply(2);                                       // 7

  // 2) Composition (∘)
  static final Function<Integer,Integer> dbl = x -> x * 2;
  static final Function<Integer,Integer> dblThenInc = inc.compose(dbl);         // inc(dbl(x))
  static final int compRes = dblThenInc.apply(10);                               // 21

  // 3) Referential Transparency
  static final int pureTotal = List.of(1,2,3).stream().reduce(0, Integer::sum); // 6
  // System.out.println("hi"); // side effect → not referentially transparent

  // 4) Immutability via records
  public record Point(int x, int y) {}
  static final Point p1 = new Point(1,1);
  static final Point p2 = new Point(2, p1.y()); // new instance; p1 unchanged

  // 5) Higher‑Order Functions
  static final int hofSum = List.of(1,2,3).stream().map(x->x*2).filter(x->x>2).reduce(0, Integer::sum);

  // 6) Functor (Optional.map)
  static final Optional<Integer> fOpt = Optional.of(42).map(x -> x + 1); // Optional[43]

  // 7) Applicative (map2): combine independent Optionals
  static <A,B,C> Optional<C> map2(Optional<A> oa, Optional<B> ob, BiFunction<A,B,C> f) {
    return oa.flatMap(a -> ob.map(b -> f.apply(a,b)));
  }
  static final Optional<String> user = map2(Optional.of("Ada"), Optional.of(36), (n,a) -> n + " is " + a);

  // 8) Monad (flatMap): dependent sequencing
  static final Optional<Integer> monadRes = Optional.of(2).flatMap(x -> Optional.of(3).map(y -> x + y));

  // 9) Natural Transformation: List<A> -> Optional<A> (head)
  static <A> Optional<A> headOption(List<A> xs) { return xs.isEmpty() ? Optional.empty() : Optional.of(xs.get(0)); }

  // 10) Monoid with reduce (associativity + identity)
  static final int mSum = List.of(1,2,3).stream().reduce(0, Integer::sum);
  static final String mStr = List.of("a","b","c").stream().reduce("", String::concat);

  // 11) ADTs & Pattern Matching (sealed + switch)
  sealed interface Shape permits Circle, Rect {}
  record Circle(double r) implements Shape {}
  record Rect(double w, double h) implements Shape {}
  static double area(Shape s) {
    return switch (s) {
      case Circle c -> Math.PI * c.r() * c.r();
      case Rect   r -> r.w() * r.h();
    };
  }

  // 12) Effects at the Edges — pure core + I/O boundary
  static int pureLogic(int x) { return x * 2; }

  public static void main(String[] args) {
    System.out.println(pureLogic(5)); // effect at boundary
  }

  // 13) Property‑Based Testing (outline with jqwik)
  // @Property
  // void functorIdentity(@ForAll List<Integer> xs){
  //   Assertions.assertEquals(xs, xs.stream().map(Function.identity()).toList());
  // }
}