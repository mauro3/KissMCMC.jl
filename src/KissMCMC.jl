module KissMCMC

export inverse_transform_sample, rejection_sample_unif, rejection_sample,
       metropolis, emcee, metropolisp, emceep, squash_chains

include("samplers-serial.jl")
include("samplers-parallel.jl")

end # module
