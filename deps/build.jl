using Libdl

const _DEPS_FILE = joinpath(dirname(@__FILE__), "deps.jl")

if isfile(_DEPS_FILE)
    rm(_DEPS_FILE)
end

function write_depsfile(path)
	open(_DEPS_FILE, "w") do f
		println(f, "const libtopicmodelsvb = \"$(escape_string(path))\"")
	end
end

if get(ENV, "JULIA_REGISTRYCI_AUTOMERGE", "false") == "true"
	write_depsfile("julia_registryci_automerge")
end
