module KissMCMC
#using StatsBase, DataFrames
using ProgressMeter, StaticArrays
#import MCMCDiagnostics

# std-lib:
using Distributed, Statistics

export metropolis, emcee, squash_chains, make_theta0s

include("samplers-serial.jl")
#include("samplers-parallel.jl")


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
            push!(cols[5], mod==nothing ? nothing : mode[i])
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
            push!(cols[4], mode==nothing ? nothing : mode[i])
            push!(cols[5], std(view(thetas,i,1:ns)))
            eff_samples!=nothing && push!(cols[6], eff_samples[i])
        end

    end
    #DataFrame(cols, header)
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
