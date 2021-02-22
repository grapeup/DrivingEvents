using Pkg

ENV["PYTHON"] = ""

pkg"activate ."
pkg"instantiate"
pkg"build PyCall"

using Conda

Conda.add("pip")

pythonbin = "python"
if Sys.iswindows()
    pythonbin = "python.exe"
end
run(`$(joinpath(Conda.PYTHONDIR, "$(pythonbin)")) -m pip install mlflow`)

pkg"precompile"
pkg"add Gadfly"
pkg"precompile"
pkg"add StatsPlots"
pkg"precompile"