ThisBuild / scalaVersion := "2.13.14"

Compile / scalaSource := baseDirectory.value / "scala"
Compile / javaSource  := baseDirectory.value / "java"

libraryDependencies += "org.typelevel" %% "cats-core" % "2.12.0"
