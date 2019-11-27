# A bunch of probability density function samplers
#
# Refs:
# https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
# https://theclevermachine.wordpress.com/2012/11/04/mcmc-multivariate-distributions-block-wise-component-wise-updates/
#
# https://pymc-devs.github.io/pymc/theory.html

"""
    metropolis(pdf, sample_ppdf, theta0;
               niter=10^5,
               nburnin=niter÷2,
               nthin=1,
               use_progress_meter=true)

Metropolis sampler, the simplest MCMC method. Can only be used with symmetric
proposal distributions (`ppdf`).

Input:

- pdf -- The probability log-density function to sample.

         It can also returns an arbitrary blob of
         something as second output argument.
         `p, blob = pdf(theta)` where `theta` are the parameters.

- sample_ppdf -- draws a sample for the proposal/jump distribution
                 `sample_ppdf(theta)`.  Needs to be symmetric:
                 `sample_ppdf(theta1)==sample_ppdf(theta2)`
- theta0 -- initial value of parameters (<:AbstractVector)

Optional keyword input:

- niter -- number of steps to take (10^5)
- nburnin -- number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- use_progress_meter=true : whether to show a progress meter

Notes:
- single threaded

Output:

- samples:
  - if typeof(theta0)==Vector then Array{eltype(theta0)}(length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array{typeof(theta0)}(niter-nburnin)
- accept_ratio: ratio accepted to total steps
- logdensities: the value of the log-density for each sample
- blobs: anything else that the pdf-function returns as second argument
"""
function metropolis(pdf, sample_ppdf, theta0;
                    niter=10^5,
                    nburnin=niter÷2,
                    nthin=1,
                    use_progress_meter=true,
                    )
    # initialize
    theta0 = deepcopy(theta0)
    pdf_ = hasblob(pdf, theta0) ? pdf : t -> (pdf(t), nothing)
    p0, blob0 = pdf_(theta0)
    prog = use_progress_meter ? Progress(length((1-nburnin):(niter-nburnin))÷nthin, 1, "Metropolis, niter=$niter: ", 25) : nothing

    # run
    #@inferred _metropolis(pdf_, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob(pdf, theta0))
    _metropolis(pdf_, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob(pdf, theta0))
end
function _metropolis(pdf, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob)
    nsamples = (niter-nburnin) ÷ nthin

    # initialize output arrays
    thetas = typeof(theta0)[];
    sizehint!(thetas, nsamples)
    blobs = typeof(blob0)[];
    hasblob && sizehint!(blobs, nsamples) # only provision if there are blobs
    logdensities = typeof(p0)[];
    sizehint!(logdensities, nsamples)

    naccept = 0
    nn = 1
    for n=(1-nburnin):(niter-nburnin)
        # take a step:
        theta1 = sample_ppdf(theta0)
        p1, blob1 = pdf(theta1)
        # if p1/p0>rand() then accept:
        if  p1-p0 > log(rand())
            theta0 = theta1
            blob0 = blob1
            p0 = p1
            naccept += 1
        end
        # storage
        if rem(n, nthin)==0
            prog!=nothing && ProgressMeter.next!(prog; showvalues =
                                                 [(:accept_ratio, round(naccept/nn, sigdigits=3)),
                                                  (:burnin_phase, n<=0)])
            if n>0 # after burnin
                push!(thetas, theta0)
                hasblob && push!(blobs, blob0)
                push!(logdensities, p0)
            end
        end
        nn +=1 # number of steps
        if n==0 # reset counters at end of burnin phase
            naccept=0
            nn=1
        end
    end
    accept_ratio = naccept/(niter-nburnin)
    return thetas, accept_ratio, logdensities, blobs
end

####
# The affine invariant MCMC sampler, aka MC hammer in its
# python emcee implementation
###
# https://github.com/dfm/emcee

"""
    emcee(pdf, theta0s;
          nchains=10^2,
          niter=10^5,
          nburnin=niter÷2,
          nthin=1,
          a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
          use_progress_meter=true)

Input:

- pdf -- The probability log-density function to sample.

         The pdf can also returns an arbitrary blob of something.
         `p, blob = pdf(theta)` where `theta` are the parameters.
- theta0s -- initial parameters, a Vector of length of the number of chains.
             Consider creating them with `make_theta0s(theta0, ballradius)`

Note that theta0s[1] need to support `.+`, `.*`, and `length`.

Optional key-word input:

- niter -- total number of steps to take (10^5) (==total number of log-density evaluations).
- nburnin -- total number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- use_progress_meter=true : whether to show a progress meter

Output:

- samples: of type & size `Matrix{typeof(theta0)}(niter-nburnin, nchains)`
- accept_ratio: ratio of accepted to total steps
- logdensities: the value of the log-density for each sample
- blobs: anything else that the pdf-function returns as second argument

Notes:
- use `squash_chains(samples)` to concatenate all chains into one chain.

Reference:
sqrt(var(naccept))
- Goodman, Jonathan, and Jonathan Weare, 2010
  http://dx.doi.org/10.2140/camcos.2010.5.65
- Foreman-Mackey et al. 2013, emcee: The MCMC hammer,
  https://github.com/dfm/emcee


Example

    thetas, accept_ratio, _, logdensities = emcee(x->sum(-x.^2),
    ([1.0,2, 5], 0.1))
"""
function emcee(pdf, theta0s;
               niter=10^5,
               nburnin=niter÷2,
               nthin=1,
               a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
               use_progress_meter=true
               )
    @assert a_scale>1
    nchains = length(theta0s)
    @assert iseven(nchains) "Use an even number of chains."
    niter_chain = niter ÷ nchains
    nburnin_chain = nburnin ÷ nchains
    @assert nchains>=length(theta0s[1])+2 "Use more chains: at least DOF+2, but better many more."

    # initialize
    pdf_ = hasblob(pdf, theta0s[1]) ? pdf : t -> (pdf(t), nothing)
    tmp = pdf_.(theta0s)
    p0s, blob0s = getindex.(tmp, 1), getindex.(tmp, 2)

    # initialize progress meter
    prog = use_progress_meter ? Progress(length((1-nburnin_chain):(niter_chain-nburnin_chain)), 1, "emcee, niter=$niter, nchains=$nchains: ", 25) : nothing
    # do the MCMC
    _emcee(pdf_, theta0s, p0s, blob0s, niter_chain, nburnin_chain, nchains, nthin, a_scale, prog, hasblob(pdf, theta0s[1]))
end

"Makes output arrays"
function init_storage(v0s, nchains, niter_chain)
    vs = [eltype(v0s)[] for i=1:nchains]
    sizehint!.(vs, niter_chain)
    return vs
end
init_storage(v0s::Array{Nothing,1}, nchains, niter_chain) = nothing

"g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
g_pdf(z, a) = 1/a<=z<=a ? 1/sqrt(z) * 1/(2*(sqrt(a)-sqrt(1/a))) : zero(z)

"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
cdf_g_inv(u, a) = (u*(sqrt(a)-sqrt(1/a)) + sqrt(1/a) )^2

"Sample from g using inverse transform sampling.  a=2.0 is recommended."
sample_g(a) = cdf_g_inv(rand(), a)

function _emcee(pdf, theta0s, p0s, blob0s, niter_chain, nburnin_chain, nchains, nthin, a_scale, prog, hasblob)
    # initialize output
    thetas = init_storage(theta0s, nchains, niter_chain)
    blobs = init_storage(blob0s, nchains, niter_chain)
    logdensities = init_storage(p0s, nchains, niter_chain)

    # initialization and work arrays:
    naccept = zeros(Int, nchains)
    N = length(theta0s[1])
    nn = 1
    for n in (1-nburnin_chain):(niter_chain-nburnin_chain)
        for batch = 1:2
            ncs, ncos = circshift(SVector(1:nchains÷2, nchains÷2+1:nchains), batch-1)
            Threads.@threads for nc in ncs
                # draw a random other chain
                no = rand(ncos)
                # sample g (eq. 10)
                z = sample_g(a_scale)

                # sample a new theta with the stretch move (eq. 7):
                theta1 = theta0s[no] .+ z .* (theta0s[nc] .- theta0s[no])
                # and calculate the theta1 density:
                p1, blob1 = pdf(theta1)

                # if z^(N-1)*p1/p0>rand() then accept:
                if (N-1)*log(z) + p1 - p0s[nc] >= log(rand())
                    theta0s[nc] = theta1
                    p0s[nc] = p1

                    blob0s[nc] = blob1
                    naccept[nc] += 1
                end

                if n>0 && rem(n,nthin)==0 # store theta after burn-in
                    push!(thetas[nc], theta0s[nc])
                    hasblob && push!(blobs[nc], blob0s[nc])
                    push!(logdensities[nc], p0s[nc])
                end
            end # for nc =1:ncs
        end # batch = 1:2
        if rem(n,nthin)==0
            macc = mean(naccept)
            sacc = sqrt(var(naccept)) # std(naccept) is not type-stable!
            outl = sum(abs.(naccept.-macc).>2*sacc)
            prog!=nothing && ProgressMeter.next!(prog;
                                                 showvalues = [(:accept_ratio_mean, round(macc/nn,sigdigits=3)),
                                                               (:accept_ratio_std, round(sacc/nn,sigdigits=3)),
                                                               (:accept_ratio_outliers, outl),
                                                               (:burnin_phase, n<=0)])
        end
        if n==0 # reset after burnin
            naccept = fill!(naccept,0)
            nn = 1
        end
        nn +=1
    end # for n=(1-nburnin_chain):(niter_chain-nburnin_chain)

    accept_ratio = [na/(niter_chain-nburnin_chain) for na in naccept]
    return thetas, accept_ratio, logdensities, blobs
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

#     # Compute the FFT and then (from that) the auto-correlation function
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


"""
    squash_chains(thetas, accept_ratio, blobs, logdensities;
                                                         drop_low_accept_ratio=false,
                                                         drop_fact=2,
                                                         verbose=true,
                                                         order=false) # if true the samples are ordered

Puts the samples of all chains into one vector.

This can also drop chains which have a low accept ratio (this can happen with
emcee, but I am anything but sure whether this is ok),
by setting drop_low_accept_ratio=true.  drop_fact -> increase to drop more chains.

Returns:
- thetas
- mean(accept_ratio[chains2keep])
- blobs
- log-densities
"""
function squash_chains(thetas, accept_ratio, logdensities, blobs;
                       drop_low_accept_ratio=false,
                       drop_fact=2,
                       verbose=true,
                       order=false)

    nchains=length(accept_ratio)
    if drop_low_accept_ratio
        chains2keep = Int[]
        # this is a bit of a hack, but emcee seems to sometimes have
        # chains which are stuck in oblivion.  These have a very low
        # acceptance ratio, thus filter them out.
        ma,sa = median(accept_ratio), std(accept_ratio)
        verbose && println("Median accept ratio is $ma, standard deviation is $sa\n")
        for nc=1:nchains
            if accept_ratio[nc]<=ma-drop_fact*sa # this 1 is heuristic
                verbose && println("Dropping chain $nc with low accept ratio $(accept_ratio[nc])")
                continue
            end
            push!(chains2keep, nc)
        end
    else
        chains2keep = collect(1:nchains)
    end

    t = copy(thetas[chains2keep[1]])
    append!.(Ref(t), thetas[chains2keep[2:end]])

    l = copy(logdensities[chains2keep[1]])
    append!.(Ref(l), logdensities[chains2keep[2:end]])

    b = if blobs==nothing
        nothing
    else
        b = copy(blobs[chains2keep])
        append!.(Ref(b), blobs[chains2keep[2:end]])
    end

    if order # order such that early samples are early in thetas
        nc = length(chains2keep)
        ns = length(thetas[1])
        perm = sortperm(vcat([collect(1:ns) for i=1:nc]...))
        if blobs!=nothing
            b = b[perm]
        end
        l = l[perm]
        t = t[perm]
    end
    return t, mean(accept_ratio[chains2keep]), l, b
end


# ####################
# """
#          retrace_samples(pdf, thetas_in;
#                          logpdf=true,
#                          blob_reduce! = default_blob_reduce!,
#                          use_progress_meter=true)

# This function allows to re-run the pdf for some, given thetas (`thetas_in`).  This is probably
# a bit niche, but hey.

# This I use when only sampling the prior but then want to get some function-blob evaluations.
# That way the evaluation of an expensive forward function can be avoided.
# """
# function retrace_samples(pdf, thetas_in;
#                          logpdf=true,
#                          blob_reduce! = default_blob_reduce!,
#                          use_progress_meter=true,
#                          )
#     # initialize
#     nchains = 0
#     niter = size(thetas_in,2)
#     nburnin = 0
#     nthin = 1
#     pdf_, p0, theta0, blob0, thetas, blobs, nchains, pdftype, logposts =
#         _initialize(pdf, thetas_in[:,1], niter, nburnin, logpdf, nchains, nthin, blob_reduce!, make_SharedArray=false)

#     prog = use_progress_meter ? Progress(length((1-nburnin):(niter-nburnin))÷nthin, 1, "Retracing samples, niter=$niter: ", 25) : nothing

#     # run
#     _retrace_samples!(p0, thetas_in, blob0, thetas, blobs, pdf_, niter,
#                       nburnin, nthin, pdftype, blob_reduce!, prog, logposts)
# end
# function _retrace_samples!(p0, thetas_in, blob0, thetas, blobs, pdf, niter,
#                            nburnin, nthin, pdftype, blob_reduce!, prog, logposts)
#     @inbounds for ni=1:size(thetas_in,2)
#         theta = thetas_in[:,ni]
#         # take a step:
#         p, blob = pdf(theta)

#         _setindex!(thetas, theta, ni)
#         blob_reduce!(blobs, blob, ni)
#         logposts[ni] = p

#         prog!=nothing && ProgressMeter.next!(prog)
#     end
#     accept_ratio = -1
#     return thetas, accept_ratio, blobs, logposts
# end
