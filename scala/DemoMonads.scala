object ScalaDemoMonads extends App {
  case class NumberWithLogs(result: Int, logs: List[String])

  def square(x: Int): NumberWithLogs =
    NumberWithLogs(x * x, List(s"Squared $x to get ${x * x}."))

  def addOne(x: NumberWithLogs): NumberWithLogs =
    NumberWithLogs(x.result + 1,
      x.logs :+ s"Added 1 to ${x.result} to get ${x.result + 1}.")

  val out = addOne(square(2))
  println(out)
}