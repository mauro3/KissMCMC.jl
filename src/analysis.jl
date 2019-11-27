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


# """
#     evaluate_convergence(thetas1, thetas2; indices=1:size(thetas)[1], walkernr=1)

# Evaluates:
# - R̂ (potential scale reduction): should be <1.1
# - total effective sample size of all chains combined, and
# - the average thinning factor

# See Gelman etal 2014, page 281-287

# Note that https://dfm.io/posts/autocorr/ says "you should not compute
# the G–R statistic using multiple chains in the same emcee ensemble because
# the chains are not independent!"  Thus this needs input from two separate emcee runs.

# Example

#     thetas1, _, accept_ratio, logposteriors = emcee(x->sum(-x.^2), ([1.0,2, 5], 0.1))
#     thetas2, _, accept_ratio, logposteriors = emcee(x->sum(-x.^2), ([1.0,2, 5], 0.1))
#     Rhat, sample_size, nthin = evaluate_convergence(thetas1, thetas2)
# """
# function evaluate_convergence(thetas... ; indices=1:size(thetas[1])[1],
#                               walkernr=1)
#     # @show [size(t) for t in thetas]
#     # @assert all([size(t)==size(thetas[1]) for t in thetas])
#     Rs = Float64[]
#     sample_size = Float64[]
#     split = size(thetas[1])[2]÷2
#     for i in indices
#         #chains = [thetas1[i,1:split,walkernr], thetas2[i,split+1:2*split,walkernr]]
#         chains = [t[i,1:split,walkernr] for t in thetas]
#         push!(Rs, MCMCDiagnostics.potential_scale_reduction(chains...))
#         push!(sample_size, sum(MCMCDiagnostics.effective_sample_size.(chains)))
#     end
#     nwalkers = length(size(thetas[1]))==3 ? size(thetas[1])[3] : 1
#     nthin = size(thetas[1])[2]*nwalkers/mean(sample_size)
#     return Rs, sample_size, isnan(nthin) ? -1 : round(Int, nthin)
# end


# """
#     int_acorr(thetas::Array{<:Any,3}; c=5, warn=true, warnat=50)

# Returns:
# - Estimated integrated autocorrelation time τ (in number of steps) for each theta
# - Indication whether the estimate has converged as ratio between chain-length and
#   τ.  Should probably be larger than 50 or 100.  (again for each theta)

# τ gives the factor by which the variance of the calculated expectation of y
# is larger than for independent samples.

# In other words:
# - τ is the number of steps that are needed before the chain "forgets" where it started.
# - N/τ is the effective number of samples (N is total number of samples)

# This means that, if you can estimate τ, then you can estimate the number
# of samples that you need to generate to reduce the relative error on your
# target integral to (say) a few percent.

# NOTE: the employed algorithm is not that great at estimating τ
# with short chain lengths.  To get good estimates a chain length greater than
# 50τ is recommended (which may well be longer than what you actually need).
# By default this function will warn if the estimate is likely not converged.

# Optionals:
# - c -- window width to search when calculating autocor (5)
# - warn -- warn if chain length is too small for an accurate τ estimate (true)

# Notes:
# - It is important to remember that f depends on the specific
#   function f(θ). This means that there isn't just one integrated
#   autocorrelation time for a given Markov chain. Instead, you must
#   compute a different τ for any integral you estimate using the samples.
# - works equally with thinned an non-thinned chains
# - whilst it is not recommended by the referenced work, it is probably ok
#   to apply this also to a squashed chain.  Then run it as:
#   `int_acorr(reshape(thetas, (size(thetas,1), size(thetas,2), 1))`

# Ref:
# https://emcee.readthedocs.io/en/latest/tutorials/autocorr/
# Adapted from various bits of the code there.
# """
# function int_acorr(thetas::Array{<:Any,3}; c=5, warn=true, warnat=50)
#     @assert c>1
#     sz = size(thetas)
#     ntheta, nsamples, nchains = size(thetas)
#     out = Float64[]
#     for n = 1:ntheta
#         # mean autocorr of all chains:
#         rho = zeros(nsamples÷2)
#         for cc = 1:nchains
#             rho .+= acor1d(thetas[n,:,cc])
#         end
#         rho ./= nchains
#         # the -1 correct see https://github.com/dfm/emcee/issues/267#issuecomment-477556521
#         taus = 2 * cumsum(rho) - 1
#         window = auto_window(taus, c)
#         push!(out, taus[window])
#     end
#     converged = nsamples./out # probably converged if > 50
#     if warn && any(converged.<warnat)
#         Base.warn("Estimate of integrated autocorrelation likely not accurate!")
#     end
#     if any(isnan.(out)) || any(isnan.(converged))
#         # set both to -1 in this case (a hack)
#         out = out*false - 1
#         converged = converged*false -1
#     end
#     return out, converged
# end

# """
#     eff_samples(thetas::Array{<:Any,3}, c=5)

# Return scalars:
# - total mean number of effective samples
# - suggested thinning step (essentially equal to the mean τ)
# - mean τ-convergence estimate  (as ratio chain-length and τ,
#   should be greater than 50 or 100 or so)
# Return vectors for each θ component
# - total number of effective samples
# - τ auto correlation steps
# - per chain estimates whether τ-estimates are converged

# (the last two output are pass on from int_acorr)

# Example:

#     Neff, τ, conv, Neffs, τs, convs = eff_samples(theta)
# """
# function eff_samples(thetas::Array{<:Any,3}; c=5)
#     acorr, converged = int_acorr(thetas, c=c, warn=false)
#     ns = size(thetas,2)./acorr * size(thetas,3)
#     return (round.(Int, mean(ns)), round.(Int, size(thetas,2)*size(thetas,3)÷mean(ns)), mean(converged),
#             round.(Int, ns), acorr, converged)
# end

# """
#     samples_vs_tau(thetas::Array{<:Any,3})

# Returns samples vs τ as is plotted in
# https://emcee.readthedocs.io/en/latest/tutorials/autocorr/

# To plot do:

#     using Plots
#     N,taus = KissMCMC.samples_vs_tau(thetas);
#     plot(N,taus)
#     # additionally plot the N/50 line
#     plot!(N, N/50, ls=:dash, c=:black)
# """
# function samples_vs_tau(thetas::Array{<:Any,3})
#     N = round(Int, logspace(2,log10(size(thetas,2)), 10))
#     nthin = 1

#     taus = []
#     # converged = []
#     NN = []
#     for n in N
#         if length(1:nthin:n)>3
#             tau, conv = KissMCMC.int_acorr(thetas[:, 1:nthin:n, :], warn=false)
#             push!(taus, tau)
#             # push!(converged, conv)
#             push!(NN,length(1:nthin:n))
#         end
#     end
#     taus = hcat(taus...)'
#     return NN, taus
# end

# """
#     error_of_estimated_mean(thetas::Array{<:Any,3},
#                             Neff = eff_samples(thetas)[4])

# Calculates the error in the estimated mean of each theta.

# Return estimates of:
# - mean(theta)
# - error of mean
# - std(theta)

# Ref:
# 15.4.3 in https://mc-stan.org/docs/2_18/reference-manual/effective-sample-size-section.html
# """
# function error_of_estimated_mean(thetas::Array{<:Any,3},
#                                  Neff = eff_samples(thetas)[4])
#     mtheta = mean(mean(thetas,2),3)[:]
#     stheta = std(std(thetas,2),3)[:]
#     err_of_mean = stheta./sqrt(Neff)
#     return mtheta, err_of_mean, stheta
# end

# ## helper functions
# import Base.DFT
# function acor1d(x::AbstractVector, norm=true)
#     # TODO for speed:
#     #n = DFT.nextprod([2,3,4,5], length(x))

#     # Compute the FFT and then (from that) """
#     hasblob(pdf, theta) = length(pdf(theta))==2

# Test if the log-pdf returns a blob
# """
# hasblob(pdf, theta) = length(pdf(theta))==2
# the auto-correlation function
#     f = DFT.fft(x - mean(x))
#     acf = real.(DFT.ifft(f .* conj(f)))# TODO [1:length(x)]
#     acf ./= 4*length(x)

#     # Optionally normalize
#     if norm
#         acf ./= acf[1]
#     end

#     return acf[1:length(acf)÷2]
# end

# """
#     auto_window(taus, c)

# Gives the ±window over which the autocorr should be integrated
# to get a good estimate.  `c` defaults to 5.
# """
# function auto_window(taus, c)
#     for (i,t) in enumerate(taus)
#         i >= c*t && return i
#     end
#     return length(taus)-1
# end
