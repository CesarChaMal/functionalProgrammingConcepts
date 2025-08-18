object ScalaDemoMonadsImproved1 extends App {
  case class NumberWithLogs(result: Int, logs: List[String])

  def wrapWithLogs(x: Int): NumberWithLogs =
    NumberWithLogs(x, List.empty)

  def runWithLogs(input: NumberWithLogs, transform: Int => NumberWithLogs): NumberWithLogs = {
    val newNumberWithLogs = transform(input.result)
    NumberWithLogs(newNumberWithLogs.result, input.logs ++ newNumberWithLogs.logs)
  }

  def square(x: Int): NumberWithLogs =
    NumberWithLogs(x * x, List(s"Squared $x to get ${x * x}."))

  def addOne(x: Int): NumberWithLogs =
    NumberWithLogs(x + 1, List(s"Added 1 to $x to get ${x + 1}."))

  def multiplyByThree(x: Int): NumberWithLogs =
    NumberWithLogs(x * 3, List(s"Multiplied $x by 3 to get ${x * 3}."))

  val a = wrapWithLogs(5)
  val b = runWithLogs(a, addOne)
  val c = runWithLogs(b, square)
  val d = runWithLogs(c, multiplyByThree)
  println(d)
}