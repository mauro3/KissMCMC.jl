module KissMCMC
using StatsBase, DataFrames
using ProgressMeter
using Compat
import Compat.view

export inverse_transform_sample, rejection_sample_unif, rejection_sample,
       metropolis, emcee, metropolisp, emceep, squash_chains,
       autocor_length, print_results

include("samplers-serial.jl")
include("samplers-parallel.jl")

if VERSION>=v"0.5-"
    # good old hist is deprecated:
    function hist(samples,nbins)
        h=fit(Histogram, samples,nbins=nbins)
        (h.edges[1], h.weights)
    end
end


######
# Evaluation
#####

"""
Auto-correlation length: Find index at which auto-correlation drops to
`corbound` (default 0.0), trys the first maxlen (default 100) lags.
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

"""
Gelman Rubin statistics

"""
function gelman_rubin(chains)
    error("not implemented")
end

"""
Calculate the mode of samples, i.e. where the pdf should be maximal.
This may not be the best way.
"""
function modehist(samples::AbstractVector, nbins=length(samples)÷10)
    hh = fit(Histogram, samples, nbins=nbins, closed=:right)
    r,h = hh.edges[1], hh.weights
    #r,h = hist(samples, nbins)
    return r[findmax(h)[2]]+step(r)/2
end

function modehist(samples::AbstractMatrix, nbins=size(samples,2)÷10)
    modes = Float64[]
    for row in 1:size(samples,1)
        push!(modes, modehist(view(samples, row, 1:size(samples,2)), nbins))
    end
    return modes
end

######
# Output
######

"Summary statistics of a run."
function summarize_run(thetas::Matrix; theta_true=similar(thetas,0), names=["$i" for i=1:size(thetas,1)])
    nt = size(thetas,1)
    ns = size(thetas,2)
    if length(theta_true)>0
        cols = Any[[], [],[],[],[],[]]
        header = [:var, :err, :median, :mean, :mode, :std]

        for i=1:nt
            push!(cols[1], Symbol(names[i]))
            t = theta_true[i]
            m = median(view(thetas,i,1:ns))
            push!(cols[2], abs(t-m))
            push!(cols[3], median(view(thetas,i,1:ns)))
            push!(cols[4], mean(view(thetas,i,1:ns)))
            push!(cols[5], modehist(view(thetas,i,1:ns)))
            push!(cols[6], std(view(thetas,i,1:ns)))
        end
    else
        cols = Any[[], [],[],[],[]]
        header = [:var, :median, :mean, :mode, :std]
        for i=1:nt
            push!(cols[1], Symbol(names[i]))
            push!(cols[2], median(view(thetas,i,1:ns)))
            push!(cols[3], mean(view(thetas,i,1:ns)))
            push!(cols[4], modehist(view(thetas,i,1:ns)))
            push!(cols[5], std(view(thetas,i,1:ns)))
        end

    end
    DataFrame(cols, header)
end

"Print result summary"
function print_results(thetas::Matrix, accept_ratio; theta_true=similar(thetas,0), names=["$i" for i=1:size(thetas,1)],
                           maxvar=45)
    println("Ratio of accepted/total steps: $accept_ratio\n")
    out = summarize_run(thetas, theta_true=theta_true, names=names)
    show(out[1:min(size(out,1),maxvar),:])
    nothing
end


end # module
