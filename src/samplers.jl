# A bunch of MCMC probability density function samplers
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
                 `ppdf(theta1)==ppdf(theta2)`.  Example:
                 `theta -> c*randn() + theta`
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
                    hasblob=false
                    )
    # initialize
    theta0 = deepcopy(theta0)
    pdf_ = hasblob ? pdf : t -> (pdf(t), nothing)
    p0, blob0 = pdf_(theta0)
    prog = use_progress_meter ? Progress(length((1-nburnin):(niter-nburnin))÷nthin, 1, "Metropolis, niter=$niter: ", 25) : nothing

    # run
    #@inferred _metropolis(pdf_, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob)
    _metropolis(pdf_, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob)
end

"Makes output arrays"
function init_output_metro(v0s, niter)
    vs = typeof(v0s)[];
    sizehint!.(vs, niter)
    return vs
end
init_output_metro(v0s::Nothing, niter) = nothing

function _metropolis(pdf, sample_ppdf, theta0, p0, blob0, niter, nburnin, nthin, prog, hasblob)
    nsamples = (niter-nburnin) ÷ nthin

    # initialize output arrays
    thetas = init_output_metro(theta0, niter)
    blobs = init_output_metro(blob0, niter)
    logdensities = init_output_metro(p0, niter)

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



"""
    emcee(pdf, theta0s;
          niter=10^5,
          nburnin=niter÷2,
          nthin=1,
          a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
          use_progress_meter=true,
          hasblob=false)

The affine invariant MCMC sampler, aka MC hammer in its Python emcee implementation


Input:

- pdf -- The probability log-density function to sample.

         The pdf can also returns an arbitrary blob of something.
         `p, blob = pdf(theta)` where `theta` are the parameters.
- theta0s -- initial parameters, a Vector of length of the number of walkers.
             Consider creating them with `make_theta0s(theta0, ballradius)`

Note that theta0s[1] need to support `.+`, `.*`, and `length`.

Optional key-word input:
- niter -- total number of steps to take (10^5) (==total number of log-density evaluations).
- nburnin -- total number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- use_progress_meter=true : whether to show a progress meter

Output:
- samples: of type & size `Matrix{typeof(theta0)}(niter-nburnin, nwalkers)`
- accept_ratio: ratio of accepted to total steps
- logdensities: the value of the log-density for each sample
- blobs: anything else that the pdf-function returns as second argument

Notes:
- use `squash_walkers(samples)` to concatenate all walkers-chains into one.

Reference:
- Goodman, Jonathan, and Jonathan Weare, 2010
  http://dx.doi.org/10.2140/camcos.2010.5.65
- emcee https://github.com/dfm/emcee
- Foreman-Mackey et al. 2013, emcee: The MCMC hammer,
  https://github.com/dfm/emcee


Example

    thetas, accept_ratio, logdensities = emcee(x->-sum(x.^2), [1.0, 2, 5, 8])
"""
function emcee(pdf, theta0s;
               niter=10^5,
               nburnin=niter÷2,
               nthin=1,
               a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
               use_progress_meter=true,
               hasblob=false
               )
    theta0s = deepcopy(theta0s)

    @assert a_scale>1
    nwalkers = length(theta0s)
    @assert iseven(nwalkers) "Use an even number of walkers."
    niter_walker = niter ÷ nwalkers
    nburnin_walker = nburnin ÷ nwalkers
    @assert nwalkers>=length(theta0s[1])+2 "Use more walkers: at least DOF+2, but better many more."

    # initialize
    pdf_ = hasblob ? pdf : t -> (pdf(t), nothing)
    tmp = pdf_.(theta0s)
    p0s, blob0s = getindex.(tmp, 1), getindex.(tmp, 2)

    # initialize progress meter
    prog = use_progress_meter ? Progress(length((1-nburnin_walker):(niter_walker-nburnin_walker)), 1, "emcee, niter=$niter, nwalkers=$nwalkers: ", 25) : nothing
    # do the MCMC
    _emcee(pdf_, theta0s, p0s, blob0s, niter_walker, nburnin_walker, nwalkers, nthin, a_scale, prog, hasblob)
end

"Makes output arrays"
function init_output_emcee(v0s, nwalkers, niter_walker)
    vs = [eltype(v0s)[] for i=1:nwalkers]
    sizehint!.(vs, niter_walker)
    return vs
end
init_output_emcee(v0s::Array{Nothing,1}, nwalkers, niter_walker) = nothing

"g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
g_pdf(z, a) = 1/a<=z<=a ? 1/sqrt(z) * 1/(2*(sqrt(a)-sqrt(1/a))) : zero(z)

"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
cdf_g_inv(u, a) = (u*(sqrt(a)-sqrt(1/a)) + sqrt(1/a) )^2

"Sample from g using inverse transform sampling.  a=2.0 is recommended."
sample_g(a) = cdf_g_inv(rand(), a)

function _emcee(pdf, theta0s, p0s, blob0s, niter_walker, nburnin_walker, nwalkers, nthin, a_scale, prog, hasblob)
    # initialize output
    thetas = init_output_emcee(theta0s, nwalkers, niter_walker)
    blobs = init_output_emcee(blob0s, nwalkers, niter_walker)
    logdensities = init_output_emcee(p0s, nwalkers, niter_walker)

    # initialization and work arrays:
    naccept = zeros(Int, nwalkers)
    N = length(theta0s[1])
    nn = 1
    for n in (1-nburnin_walker):(niter_walker-nburnin_walker)
        for batch = 1:2
            ncs, ncos = circshift(SVector(1:nwalkers÷2, nwalkers÷2+1:nwalkers), batch-1)
            Threads.@threads for nc in ncs
                # draw a random other walker
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
    end # for n=(1-nburnin_walker):(niter_walker-nburnin_walker)

    accept_ratio = [na/(niter_walker-nburnin_walker) for na in naccept]
    return thetas, accept_ratio, logdensities, blobs
end


"""
    make_theta0s(theta0::T, ball_radius::T, pdf, nwalkers;
                      ball_radius_halfing_steps=7, # make the ball smaller this often
                      ntries=100*nwalkers, # how many tries per pall radius
                      hasblob=false) where T

Tries to find theta0s with nonzero density within a ball from theta0.  To be used with
`emcee`.

Example

    pdf = x-> -sum(x.^2)
    nwalkers = 100
    samples = emcee(pdf, make_theta0s([0.0, 0.0], [0.1, 0.1], pdf, nwalkers), nburnin=0, use_progress_meter=false);
"""
function make_theta0s(theta0::T, ball_radius, pdf, nwalkers;
                      ball_radius_halfing_steps=7,
                      ntries=100,
                      hasblob=false) where T
    npara = length(theta0)
    if ball_radius isa Number && !(T<:Number)
        ball_radius = ones(npara) * ball_radius
    end
    @assert length(ball_radius)==npara

    theta0s = T[]

    for i=1:nwalkers
        for k=1:ball_radius_halfing_steps
            j = 0
            ball_radius *= 1/2^(k-1)
            for j=1:ntries
                tmp = if npara==1
                    theta0 .+ randn().*ball_radius
                else
                    theta0 .+ randn(npara).*ball_radius
                end
                if hasblob
                    p0, blob0 = pdf(tmp)
                else
                    p0, blob0 = pdf(tmp), nothing
                end
                if p0>-Inf
                    push!(theta0s, tmp)
                    break
                end
            end
            length(theta0s)==i && break # found suitable thetas for this walker
            j==ntries && k==ball_radius_halfing_steps &&
                error("Could not find suitable initial theta.  PDF is zero in too many places inside ball.")
        end
    end
    return theta0s
end


"""
    squash_walkers(thetas, accept_ratio, blobs, logdensities;
                                                         drop_low_accept_ratio=false,
                                                         drop_fact=2,
                                                         verbose=true,
                                                         order=false) # if true the samples are ordered

Puts the samples of all emcee walkers into one vector.

This can also drop walkers which have a low accept ratio (this can happen with
emcee, but I am anything but sure whether this is ok),
by setting drop_low_accept_ratio=true.  drop_fact -> increase to drop more walkers.

Returns:
- thetas
- mean(accept_ratio[walkers2keep])
- log-densities
- blobs
"""
function squash_walkers(thetas, accept_ratio, logdensities=nothing, blobs=nothing;
                       drop_low_accept_ratio=false,
                       drop_fact=2,
                       verbose=true,
                       order=false)

    nwalkers=length(accept_ratio)
    if drop_low_accept_ratio
        walkers2keep = Int[]
        # this is a bit of a hack, but emcee seems to sometimes have
        # walkers which are stuck in oblivion.  These have a very low
        # acceptance ratio, thus filter them out.
        ma,sa = median(accept_ratio), std(accept_ratio)
        verbose && println("Median accept ratio is $ma, standard deviation is $sa\n")
        for nc=1:nwalkers
            if accept_ratio[nc]<=ma-drop_fact*sa # this 1 is heuristic
                verbose && println("Dropping walker $nc with low accept ratio $(accept_ratio[nc])")
                continue
            end
            push!(walkers2keep, nc)
        end
    else
        walkers2keep = collect(1:nwalkers)
    end

    t = copy(thetas[walkers2keep[1]])
    append!.(Ref(t), thetas[walkers2keep[2:end]])

    if logdensities==nothing
        l = nothing
    else
        l = copy(logdensities[walkers2keep[1]])
        append!.(Ref(l), logdensities[walkers2keep[2:end]])
    end

    if blobs==nothing
        b = nothing
    else
        b = copy(blobs[walkers2keep])
        append!.(Ref(b), blobs[walkers2keep[2:end]])
    end

    if order # order such that early samples are early in thetas
        nc = length(walkers2keep)
        ns = length(thetas[1])
        perm = sortperm(vcat([collect(1:ns) for i=1:nc]...))
        if b!=nothing
            b = b[perm]
        end
        if l!=nothing
            l = l[perm]
        end
        t = t[perm]
    end
    return t, mean(accept_ratio[walkers2keep]), l, b
end

## TODO
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
#     nwalkers = 0
#     niter = size(thetas_in,2)
#     nburnin = 0
#     nthin = 1
#     pdf_, p0, theta0, blob0, thetas, blobs, nwalkers, pdftype, logposts =
#         _initialize(pdf, thetas_in[:,1], niter, nburnin, logpdf, nwalkers, nthin, blob_reduce!, make_SharedArray=false)

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
