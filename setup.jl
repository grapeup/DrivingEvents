using Pkg

ENV["PYTHON"] = ""

pkg"activate ."
pkg"instantiate"
pkg"build PyCall"

using Conda

Conda.add("pip")

Conda.add_channel("conda-forge")
Conda.add("poppler")

pythonbin = "python"
if Sys.iswindows()
    pythonbin = "python.exe"
end
run(`$(joinpath(Conda.PYTHONDIR, "$(pythonbin)")) -m pip mlflow`)

pkg"precompile"
