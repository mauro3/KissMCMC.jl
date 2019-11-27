module KissMCMC

using ProgressMeter, StaticArrays

# std-lib:
using Statistics

export metropolis, emcee, squash_chains, make_theta0s

include("samplers.jl")
include("analysis.jl")

end # module
