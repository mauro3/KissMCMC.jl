module KissMCMC
using StatsBase, DataFrames
using ProgressMeter
using Compat
import Compat.view

export inverse_transform_sample, rejection_sample_unif, rejection_sample,
       metropolis, emcee, metropolisp, emceep, squash_chains,
       autocor_length, print_results, int_acorr, eff_samples

include("samplers-serial.jl")
include("samplers-parallel.jl")

if VERSION>=v"0.5-"
    # good old hist is deprecated:
    function hist(samples,nbins)
        h=fit(Histogram, samples,nbins=nbins)
        (h.edges[1], h.weights)
    end
end


hasblob(pdf, theta) = length(pdf(theta))==2
hasblob(pdf, theta::Union{Tuple,AbstractVector{<:AbstractVector}}) = length(pdf(theta[1]))==2

######
# Evaluation
#####


"""
Calculate the mode of samples, i.e. where the pdf should be maximal.


Note: This may not be the best way.  In fact, it is dog-slow for large data.
"""
function modehist(samples::AbstractVector, nbins=length(samples)÷10)
    hh = fit(Histogram, samples, nbins=nbins, closed=:right)
    r,h = hh.edges[1], hh.weights
    #r,h = hist(samples, nbins)
    return r[findmax(h)[2]]+step(r)/2
end

function modehist(samples::AbstractMatrix, nbins=size(samples,2)÷10)
    error("This does not work...")
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
function summarize_run(thetas::Matrix; theta_true=similar(thetas,0), names=["$i" for i=1:size(thetas,1)],
                       eff_samples=nothing, mode=nothing)
    nt = size(thetas,1)
    ns = size(thetas,2)
    if length(theta_true)>0
        cols = eff_samples==nothing ? Any[[], [],[],[],[],[]] : Any[[], [],[],[],[],[],[]]
        header = eff_samples==nothing ? [:var, :err, :median, :mean, :mode, :std] : [:var, :err, :median, :mean, :mode, :std, :eff_samples]

        for i=1:nt
            push!(cols[1], Symbol(names[i]))
            t = theta_true[i]
            m = median(view(thetas,i,1:ns))
            push!(cols[2], abs(t-m))
            push!(cols[3], median(view(thetas,i,1:ns)))
            push!(cols[4], mean(view(thetas,i,1:ns)))
            push!(cols[5], mod==nothing? nothing : mode[i])
            push!(cols[6], std(view(thetas,i,1:ns)))
            eff_samples!=nothing && push!(cols[7], eff_samples[i])
        end
    else
        cols = eff_samples==nothing ? Any[[], [],[],[],[]] : Any[[], [],[],[],[], []]
        header = eff_samples==nothing ? [:var, :median, :mean, :mode, :std] : [:var, :median, :mean, :mode, :std, :eff_samples]
        for i=1:nt
            push!(cols[1], Symbol(names[i]))
            push!(cols[2], median(view(thetas,i,1:ns)))
            push!(cols[3], mean(view(thetas,i,1:ns)))
            push!(cols[4], mode==nothing? nothing : mode[i])
            push!(cols[5], std(view(thetas,i,1:ns)))
            eff_samples!=nothing && push!(cols[6], eff_samples[i])
        end

    end
    DataFrame(cols, header)
end

"Print result summary"
function print_results(thetas::Matrix, accept_ratio, eff_samples=nothing, mode=nothing; theta_true=similar(thetas,0),
                       names=["$i" for i=1:size(thetas,1)],
                       maxvar=45,
                       title="")
    println(title)
    println("Ratio of accepted/total steps: $accept_ratio\n")
    out = summarize_run(thetas, theta_true=theta_true, names=names, eff_samples=eff_samples)
    show(out[1:min(size(out,1),maxvar),:])
    nothing
end


end # module
