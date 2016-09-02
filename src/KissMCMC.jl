module KissMCMC
using StatsBase
using ProgressMeter
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
    r,h = hist(samples, nbins)
    return r[findmax(h)[2]]
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

"Print result summary"
function print_results(thetas::Matrix, accept_ratio; title="", theta_true=similar(thetas,0), names=["$i" for i=1:size(thetas,1)],
                       prec=2, maxvar=45)
    nt = size(thetas,1)
    ns = size(thetas,2)
    io = IOBuffer()
    println(io, title)
    println(io, "Ratio of accepted/total steps: $accept_ratio\n")
    if length(theta_true)>0
        println(io,"var \t err\tmedian\t mean \t mode \t std")
        for i=1:min(maxvar,nt)
            n = names[i]
            t = round(theta_true[i],prec)
            m = round(median(view(thetas,i,1:ns)),prec)
            err = round(abs(t-m),prec)
            me  = round(mean(view(thetas,i,1:ns)),prec)
            mo  = round(modehist(view(thetas,i,1:ns)),prec)
            s = round(std(view(thetas,i,1:ns)),prec)
            println(io, "$n \t $err \t $m \t $me \t $mo \t $s")
        end
    else
        println(io,"var\tmedian \t mean \t mode \t std")
        for i=1:min(maxvar,nt)
            n = names[i]
            m = round(median(view(thetas,i,1:ns)),prec)
            me  = round(mean(view(thetas,i,1:ns)),prec)
            mo  = round(modehist(view(thetas,i,1:ns)),prec)
            s = round(std(view(thetas,i,1:ns)),prec)
            println(io, "$n \t $m \t $me \t $mo \t $s")
        end
    end
    if maxvar<nt
        print(io,"...")
    end
    println(io, "")
    print(takebuf_string(io))
end


end # module
