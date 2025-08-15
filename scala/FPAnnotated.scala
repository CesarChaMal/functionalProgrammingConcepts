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

object FPAnnotated extends App {
  // 1) Lambda, Application, Currying, Partial Application
  val inc: Int => Int = x => x + 1                  // λx. x + 1
  val add: Int => Int => Int = x => y => x + y      // λx. λy. x + y (curried)
  val add5: Int => Int = add(5)                     // partial application
  val seven: Int = add5(2)                          // application => 7

  // 2) Composition (∘): compose f after g to avoid temporaries
  def compose[A,B,C](f: B => C, g: A => B): A => C = a => f(g(a))
  val double: Int => Int = _ * 2
  val compRes: Int = compose(inc, double)(10)       // 21 = inc(double(10))

  // 3) Referential Transparency: pure expressions can be replaced with their values
  val pureTotal: Int = List(1,2,3).sum              // == 6; safe replacement
  // println("hi") // I/O is a side effect → not referentially transparent

  // 4) Immutability: new values, no mutation
  final case class Point(x: Int, y: Int)
  val p1 = Point(1,1)
  val p2 = p1.copy(x = 2)                           // p1 unchanged

  // 5) Higher‑Order Functions: map/filter/reduce composition
  val hofSum: Int = List(1,2,3).map(_*2).filter(_>2).reduce(_+_) // 10

  // 6) Functor: map preserves structure (laws: identity, composition)
  val fList = List(1,2,3).map(_ + 1)                // List(2,3,4)
  val fOpt  = Option(42).map(_ + 1)                 // Some(43)

  // 7) Applicative: combine independent Option values
  val name: Option[String] = Some("Ada")
  val age:  Option[Int]    = Some(36)
  val user: Option[(String, Int)] = (name, age).tupled           // Some(("Ada",36))
  val userStr: Option[String] = (name, age).mapN((n,a) => s"$n is $a")

  // 8) Monad: sequence dependent computations with flatMap / for‑comprehension
  val monadRes: Option[Int] = for { x <- Option(2); y <- Option(3) } yield x + y   // Some(5)

  // 9) Natural Transformation: List ~> Option (headOption)
  def headOption[A](xs: List[A]): Option[A] = xs.headOption
  val nat: Option[Int] = headOption(List(10, 20))                   // Some(10)

  // 10) Monoid: associative op + identity; Cats supplies instances
  val mSum: Int = List(1,2,3).combineAll                            // 6 (Int monoid is +/0)
  val mStr: String = List("a","b","c").combineAll                 // "abc" (concat/"")

  // 11) ADTs & Pattern Matching: encode domain and handle exhaustively
  sealed trait Shape
  final case class Circle(r: Double) extends Shape
  final case class Rect(w: Double, h: Double) extends Shape
  def area(s: Shape): Double = s match {
    case Circle(r) => Math.PI * r * r
    case Rect(w,h) => w * h
  }

  // 12) Effects at the Edges: pure core + I/O boundary
  object Domain { def pureLogic(x: Int): Int = x * 2 }
  println(Domain.pureLogic(5)) // side effect only at boundary

  // 13) Property‑Based Testing (outline): Functor identity/composition
  // ScalaCheck style pseudo:
  // forAll { (xs: List[Int]) => xs.map(identity) == xs }
  // forAll { (xs: List[Int], f: Int=>Int, g: Int=>Int) =>
  //   xs.map(f compose g) == xs.map(g).map(f)
  // }
}