# setup_scala.sh
#!/usr/bin/env bash
set -euo pipefail

# Create/overwrite build.sbt (Scala + Java custom source roots)
cat > build.sbt <<'EOF'
ThisBuild / scalaVersion := "2.13.14"

Compile / scalaSource := baseDirectory.value / "scala"
Compile / javaSource  := baseDirectory.value / "java"

libraryDependencies += "org.typelevel" %% "cats-core" % "2.12.0"
EOF

# SBT version
cat > build.properties <<'EOF'
sbt.version=1.10.2
#!/usr/bin/env bash
# setup_scala.sh
set -euo pipefail

# Create/overwrite build.sbt (uses root scala & java directories)
cat > build.sbt <<'EOF'
ThisBuild / scalaVersion := "2.13.14"

Compile / scalaSource := baseDirectory.value / "scala"
Compile / javaSource  := baseDirectory.value / "java"

libraryDependencies += "org.typelevel" %% "cats-core" % "2.12.0"
EOF

# Mandatory: sbt build properties inside project/
mkdir -p project
cat > project/build.properties <<'EOF'
sbt.version=1.10.2
EOF

# Optional: .gitignore additions
if ! grep -q '^target/$' .gitignore 2>/dev/null; then
  printf "target/\nproject/target/\nproject/project/\nout/\n" >> .gitignore
fi

# Clean previous build outputs
rm -rf target out project/target project/project

# Compile
sbt -batch clean reload evicted compile

# Quick Cats smoke test
echo 'import cats._, cats.implicits._; println((Option(1),Option(2)).tupled); println(List("a","b","c").combineAll); sys.exit(0)' | sbt console

# Optional: ensure git ignores build outputs
if ! grep -q '^target/$' .gitignore 2>/dev/null; then
  printf "target/\nproject/target/\nproject/project/\nout/\n" >> .gitignore
fi

# Clean old build outputs
rm -rf target out project/target project/project

# Compile (non-interactive)
sbt -batch clean reload evicted compile

# Quick Cats test (prints Some((1,2)) and abc)
echo 'import cats._, cats.implicits._; println((Option(1),Option(2)).tupled); println(List("a","b","c").combineAll); sys.exit(0)' | sbt console
