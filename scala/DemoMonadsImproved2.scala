object ScalaDemoMonadsImproved2 extends App {
  case class NumberWithLogs(result: Int, logs: List[String]) {
    def flatMap(f: Int => NumberWithLogs): NumberWithLogs = {
      val next = f(result)
      NumberWithLogs(next.result, logs ++ next.logs)
    }
  }

  object NumberWithLogs {
    def pure(x: Int): NumberWithLogs = NumberWithLogs(x, Nil)
    def apply(result: Int, log: String): NumberWithLogs = NumberWithLogs(result, List(log))
  }

  def square(x: Int): NumberWithLogs =
    NumberWithLogs(x * x, s"Squared $x to get ${x * x}.")

  def addOne(x: Int): NumberWithLogs =
    NumberWithLogs(x + 1, s"Added 1 to $x to get ${x + 1}.")

  def multiplyByThree(x: Int): NumberWithLogs =
    NumberWithLogs(x * 3, s"Multiplied $x by 3 to get ${x * 3}.")

  val result = NumberWithLogs.pure(5)
    .flatMap(addOne)
    .flatMap(square)
    .flatMap(multiplyByThree)
    
  println(result)
}