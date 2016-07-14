module KissMCMC
using StatsBase

export inverse_transform_sample, rejection_sample_unif, rejection_sample,
       metropolis, emcee, metropolisp, emceep, squash_chains,
       autocor_length

include("samplers-serial.jl")
include("samplers-parallel.jl")

######
# Evaluation
#####

"""
Find index at which auto-correlation drops to `corbound` (default
0.0), trys the first maxlen (default 100) lags.
"""
function autocor_length(samples::AbstractVector, maxlen=100, corbound=0.0)
    lag = 1
    while autocor(samples,[lag])[1]>corbound && lag<maxlen
        lag +=1
    end
    return lag
end
function  autocor_length(samples::AbstractMatrix, maxlen=100, corbound=0.0)
    len = maxlen
    for row in 1:size(samples,1)
        # len = min(len, autocor_length(view(samples, row, 1:size(samples,2))), maxlen, corbound)
        len = min(len, autocor_length(vec(samples[row,:]), maxlen, corbound) )
    end
    return len
end


end # module
