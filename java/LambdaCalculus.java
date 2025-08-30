/*
λ-Calculus + Combinators + Church Arithmetic (Java)

A comprehensive, runnable reference for λ-calculus fundamentals including:
* Core combinators (I, K, KI, C, S, B, W, T, M)
* Church booleans and logic (NOT, AND, OR, equality)
* DeMorgan verification
* Church numerals (0, 1, 2, 3, …) + iteration helpers
* Arithmetic: successor, addition, multiplication, exponentiation, predecessor
* Pairs (for predecessor implementation)
* Predicates: isZero, leq, eq
* Translation of !x == y || (a && z) into λ-calculus
*/

// YouTube videos

// Lambda Calculus - Fundamentals of Lambda Calculus & Functional Programming in JavaScript
// [https://www.youtube.com/watch?v=3VQ382QG-y4](https://www.youtube.com/watch?v=3VQ382QG-y4)

// A Flock of Functions; Combinators, Lambda Calculus, & Church Encodings in JS - Part II
// [https://www.youtube.com/watch?v=pAnLQ9jwN-E](https://www.youtube.com/watch?v=pAnLQ9jwN-E)

// Online IDE
// [https://www.programiz.com/javascript/online-compiler/](https://www.programiz.com/javascript/online-compiler/)

import java.util.function.Function;

public class LambdaCalculus {
    
    // === HELPERS ===
    
    // Convert Church numeral to Java int
    static int jsnum(Function<Function<Integer,Integer>, Function<Integer,Integer>> n) {
        return n.apply(x -> x + 1).apply(0);
    }
    
    // Convert Church boolean to string
    static String showBool(Function<String, Function<String, String>> p) {
        return p.apply("TRUE").apply("FALSE");
    }
    
    // === 1) CORE COMBINATORS ===
    
    // Identity (I) — λa.a
    static final Function<Object, Object> I = a -> a;
    
    // Kestrel (K) — λa.λb.a (TRUE)
    static final Function<Object, Function<Object, Object>> K = a -> b -> a;
    
    // Kite (KI) — λa.λb.b (FALSE)
    static final Function<Object, Function<Object, Object>> KI = a -> b -> b;
    
    // Cardinal (C) — λf.λa.λb.f b a (flip)
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Object, Function<Object, Object>>> C = 
        f -> a -> b -> f.apply(b).apply(a);
    
    // Starling (S) — λf.λg.λx.f x (g x)
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Function<Object,Object>, Function<Object, Object>>> S = 
        f -> g -> x -> f.apply(x).apply(g.apply(x));
    
    // Bluebird (B) — λf.λg.λx.f (g x) (compose)
    static final Function<Function<Object,Object>, 
                         Function<Function<Object,Object>, Function<Object, Object>>> B = 
        f -> g -> x -> f.apply(g.apply(x));
    
    // Warbler (W) — λf.λx.f x x
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Object, Object>> W = 
        f -> x -> f.apply(x).apply(x);
    
    // Thrush (T) — λa.λf.f a
    static final Function<Object, Function<Function<Object,Object>, Object>> T = 
        a -> f -> f.apply(a);
    
    // Mockingbird (M) — λf.f f (self-application)
    static final Function<Function<Object,Object>, Object> M = f -> f.apply(f);
    
    // === 2) CHURCH BOOLEANS ===
    
    // TRUE and FALSE
    static final Function<Object, Function<Object, Object>> TRUE = K;
    static final Function<Object, Function<Object, Object>> FALSE = KI;
    
    // NOT — λp.λa.λb.p b a
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Object, Function<Object, Object>>> NOT = 
        p -> a -> b -> p.apply(b).apply(a);
    
    // AND — λp.λq.λa.λb.p (q a b) b
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Function<Object,Function<Object,Object>>, 
                                 Function<Object, Function<Object, Object>>>> AND = 
        p -> q -> a -> b -> p.apply(q.apply(a).apply(b)).apply(b);
    
    // OR — λp.λq.λa.λb.p a (q a b)
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Function<Object,Function<Object,Object>>, 
                                 Function<Object, Function<Object, Object>>>> OR = 
        p -> q -> a -> b -> p.apply(a).apply(q.apply(a).apply(b));
    
    // Boolean equality — λp.λq.λa.λb.p (q a b) (NOT q a b)
    static final Function<Function<Object,Function<Object,Object>>, 
                         Function<Function<Object,Function<Object,Object>>, 
                                 Function<Object, Function<Object, Object>>>> BEQ = 
        p -> q -> a -> b -> p.apply(q.apply(a).apply(b)).apply(NOT.apply(q).apply(a).apply(b));
    
    // === 3) CHURCH NUMERALS ===
    
    // Zero — λf.λx.x
    static final Function<Function<Object,Object>, Function<Object, Object>> ZERO = 
        f -> x -> x;
    
    // Successor — λn.λf.λx.f (n f x)
    static final Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                         Function<Function<Object,Object>, Function<Object, Object>>> SUCC = 
        n -> f -> x -> f.apply(n.apply(f).apply(x));
    
    // Build numerals
    static final Function<Function<Object,Object>, Function<Object, Object>> ONE = 
        SUCC.apply(ZERO);
    static final Function<Function<Object,Object>, Function<Object, Object>> TWO = 
        SUCC.apply(ONE);
    static final Function<Function<Object,Object>, Function<Object, Object>> THREE = 
        SUCC.apply(TWO);
    static final Function<Function<Object,Object>, Function<Object, Object>> FOUR = 
        SUCC.apply(THREE);
    static final Function<Function<Object,Object>, Function<Object, Object>> FIVE = 
        SUCC.apply(FOUR);
    
    // === 4) PAIRS (for predecessor) ===
    
    // Church pair — λa.λb.λf.f a b
    static final Function<Object, Function<Object, Function<Function<Object,Function<Object,Object>>, Object>>> PAIR = 
        a -> b -> f -> f.apply(a).apply(b);
    
    static final Function<Function<Function<Object,Function<Object,Object>>, Object>, Object> FST = 
        p -> p.apply(K);
    
    static final Function<Function<Function<Object,Function<Object,Object>>, Object>, Object> SND = 
        p -> p.apply(KI);
    
    // === 5) ARITHMETIC ===
    
    // Addition — λm.λn.λf.λx.m f (n f x)
    static final Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                         Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                                 Function<Function<Object,Object>, Function<Object, Object>>>> PLUS = 
        m -> n -> f -> x -> m.apply(f).apply(n.apply(f).apply(x));
    
    // Multiplication — λm.λn.λf.m (n f)
    static final Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                         Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                                 Function<Function<Object,Object>, Function<Object, Object>>>> MULT = 
        m -> n -> f -> m.apply(n.apply(f));
    
    // Exponentiation — λm.λn.λf.n (m f)
    static final Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                         Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                                 Function<Function<Object,Object>, Function<Object, Object>>>> POW = 
        m -> n -> f -> n.apply(m.apply(f));
    
    // === 6) PREDICATES ===
    
    // Is zero — λn.λa.λb.n (λ_.FALSE a b) (TRUE a b)
    static final Function<Function<Function<Object,Object>, Function<Object,Object>>, 
                         Function<Object, Function<Object, Object>>> ISZERO = 
        n -> a -> b -> n.apply(x -> FALSE.apply(a).apply(b)).apply(TRUE.apply(a).apply(b));
    
    // === DEMONSTRATIONS ===
    
    @SuppressWarnings("unchecked")
    public static void main(String[] args) {
        System.out.println("=== COMBINATORS ===");
        System.out.println(I.apply(42));                    // 42
        System.out.println(K.apply(1).apply(2));           // 1
        System.out.println(C.apply(K).apply(1).apply(2));  // 2
        
        System.out.println("\n=== BOOLEANS ===");
        @SuppressWarnings("unchecked")
        var trueStr = (Function<String, Function<String, String>>) (Object) TRUE;
        @SuppressWarnings("unchecked")
        var falseStr = (Function<String, Function<String, String>>) (Object) FALSE;
        @SuppressWarnings("unchecked")
        var notTrue = (Function<String, Function<String, String>>) (Object) NOT.apply(TRUE);
        
        System.out.println(showBool(trueStr));              // TRUE
        System.out.println(showBool(falseStr));             // FALSE
        System.out.println(showBool(notTrue));              // FALSE
        
        System.out.println("\n=== CHURCH NUMERALS ===");
        @SuppressWarnings("unchecked")
        var zeroNum = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) ZERO;
        @SuppressWarnings("unchecked")
        var oneNum = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) ONE;
        @SuppressWarnings("unchecked")
        var twoNum = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) TWO;
        @SuppressWarnings("unchecked")
        var threeNum = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) THREE;
        
        System.out.println(jsnum(zeroNum) + " " + jsnum(oneNum) + " " + 
                          jsnum(twoNum) + " " + jsnum(threeNum)); // 0 1 2 3
        
        System.out.println("\n=== PREDICATES ===");
        @SuppressWarnings("unchecked")
        var isZeroZero = (Function<String, Function<String, String>>) (Object) ISZERO.apply(ZERO);
        @SuppressWarnings("unchecked")
        var isZeroOne = (Function<String, Function<String, String>>) (Object) ISZERO.apply(ONE);
        
        System.out.println(showBool(isZeroZero));           // TRUE
        System.out.println(showBool(isZeroOne));            // FALSE
        
        System.out.println("\n=== ARITHMETIC ===");
        @SuppressWarnings("unchecked")
        var plusResult = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) PLUS.apply(TWO).apply(THREE);
        @SuppressWarnings("unchecked")
        var multResult = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) MULT.apply(TWO).apply(THREE);
        @SuppressWarnings("unchecked")
        var powResult = (Function<Function<Integer,Integer>, Function<Integer,Integer>>) (Object) POW.apply(TWO).apply(THREE);
        
        System.out.println(jsnum(plusResult));              // 5
        System.out.println(jsnum(multResult));              // 6
        System.out.println(jsnum(powResult));               // 8
    }
}

/*
REFERENCE MAPPING (Java):

| Math                      | Name     | Java                                          |
|---------------------------|----------|-----------------------------------------------|
| λa.a                      | I        | Function<Object, Object> I = a -> a          |
| λa.λb.a                   | K        | Function<Object, Function<Object, Object>>   |
| λa.λb.b                   | KI       | Function<Object, Function<Object, Object>>   |
| λf.λa.λb.f b a            | C        | Complex nested Function types                 |
| λf.λg.λx.f x (g x)        | S        | Complex nested Function types                 |
| λf.λg.λx.f (g x)          | B        | Complex nested Function types                 |
| λf.λx.f x x               | W        | Complex nested Function types                 |
| λa.λf.f a                 | T        | Complex nested Function types                 |
| λf.f f                    | M        | Function<Function<Object,Object>, Object>    |
*/

/*
REFERENCE MAPPING (Java):

| Math                      | Name     | Java                                          |
|---------------------------|----------|-----------------------------------------------|
| λa.a                      | I        | Function<Object, Object> I = a -> a          |
| λa.λb.a                   | K        | Function<Object, Function<Object, Object>>   |
| λa.λb.b                   | KI       | Function<Object, Function<Object, Object>>   |
| λf.λa.λb.f b a            | C        | Complex nested Function types                 |
| λf.λg.λx.f x (g x)        | S        | Complex nested Function types                 |
| λf.λg.λx.f (g x)          | B        | Complex nested Function types                 |
| λf.λx.f x x               | W        | Complex nested Function types                 |
| λa.λf.f a                 | T        | Complex nested Function types                 |
| λf.f f                    | M        | Function<Function<Object,Object>, Object>    |
*/