/*
 FP CONCEPTS (TypeScript) — ALL IN ONE, ANNOTATED
 1) Lambda, Application, Currying, Partial Application — lambdas and closures.
 2) Composition (∘) — compose(f, g)(x) == f(g(x)).
 3) Referential Transparency — pure expressions can be substituted by values.
 4) Immutability — prefer new objects (spreads) vs mutation.
 5) Higher‑Order Functions — array map/filter/reduce pipelines.
 6) Functor — Array.map, Option.map (custom).
 7) Applicative — ap / liftA2 for Option (independent combination).
 8) Monad — flatMap for Option; Promise.then is also a Monad.
 9) Natural Transformation — Array<A> -> Option<A> (head).
 10) Monoid — reduce with associative op and identity.
 11) ADTs & Pattern Matching — discriminated unions + exhaustive handling.
 12) Effects at the Edges — keep core pure; do I/O at boundary.
 13) Property‑Based Testing — outline with fast-check.
*/

// 1) Lambda, Application, Currying, Partial Application
const inc = (x: number) => x + 1;
const add = (x: number) => (y: number) => x + y; // curried
const add5 = add(5);                              // partial
const seven = add5(2);                            // 7

// 2) Composition (∘)
const compose = <A,B,C>(f: (b:B)=>C, g:(a:A)=>B) => (a:A) => f(g(a));
const double = (x: number) => x * 2;
const compRes = compose(inc, double)(10);         // 21

// 3) Referential Transparency
const pureTotal = [1,2,3].reduce((a,b)=>a+b, 0);  // == 6
// console.log("hi"); // I/O → not referentially transparent

// 4) Immutability
type Point = Readonly<{ x: number; y: number }>;
const p1: Point = { x: 1, y: 1 };
const p2: Point = { ...p1, x: 2 };                // new object; p1 unchanged

// 5) Higher‑Order Functions
const hofSum = [1,2,3].map(x=>x*2).filter(x=>x>2).reduce((a,b)=>a+b, 0); // 10

// Option helpers (for 6–9)
type Option<T> = { _tag: 'Some'; value: T } | { _tag: 'None' };
const Some = <T>(value: T): Option<T> => ({ _tag: 'Some', value });
const None: Option<never> = { _tag: 'None' };
const isSome = <T>(o: Option<T>): o is { _tag: 'Some'; value: T } => o._tag === 'Some';

// 6) Functor: map preserves structure
const map = <A,B>(oa: Option<A>, f: (a: A) => B): Option<B> => isSome(oa) ? Some(f(oa.value)) : None;
const fArr = [1,2,3].map(x => x+1);               // [2,3,4]
const fOpt = map(Some(42), x => x + 1);           // Some(43)

// 7) Applicative: ap / liftA2 combine independent contexts
const ap = <A,B>(of: Option<(a:A)=>B>, oa: Option<A>): Option<B> => isSome(of) && isSome(oa) ? Some(of.value(oa.value)) : None;
const pure = <T>(x: T): Option<T> => Some(x);
const liftA2 = <A,B,C>(f: (a:A, b:B)=>C) => (oa: Option<A>) => (ob: Option<B>) => ap(ap(pure((a:A)=> (b:B)=> f(a,b)), oa), ob);
const name: Option<string> = Some('Ada');
const age:  Option<number> = Some(36);
const user = liftA2((n:string, a:number) => ({ n, a }))(name)(age); // Some({n:'Ada',a:36})

// 8) Monad: flatMap for dependent sequencing; Promise.then also monadic
const flatMap = <A,B>(oa: Option<A>, f: (a:A) => Option<B>): Option<B> => isSome(oa) ? f(oa.value) : None;
const monadRes = flatMap(Some(2), x => map(Some(3), y => x + y));     // Some(5)
// Promise.resolve(2).then(x => Promise.resolve(3).then(y => x+y));

// 9) Natural Transformation — Array<A> -> Option<A>
const headOption = <A>(xs: A[]): Option<A> => xs.length ? Some(xs[0]) : None;

// 10) Monoid — reduce with associative op and identity
const mSum = [1,2,3].reduce((a,b)=>a+b, 0);
const mStr = ['a','b','c'].reduce((a,b)=>a+b, '');

// 11) ADTs & Pattern Matching
type Shape = { tag: 'Circle'; r: number } | { tag: 'Rect'; w: number; h: number };
const area = (s: Shape): number => s.tag === 'Circle' ? Math.PI * s.r * s.r : s.w * s.h;

// 12) Effects at the Edges — pure core + I/O boundary
const pureLogic = (x: number) => x * 2;
function mainIO(){ console.log(pureLogic(5)); }

// 13) Property‑Based Testing (outline with fast-check)
// import * as fc from 'fast-check';
// fc.assert(fc.property(fc.array(fc.integer()), xs => xs.map(x=>x).every((v,i) => v === xs[i])));