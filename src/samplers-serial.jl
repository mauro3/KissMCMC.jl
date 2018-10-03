# A bunch of probability density function samplers
#
# Refs:
# https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
# https://theclevermachine.wordpress.com/2012/11/04/mcmc-multivariate-distributions-block-wise-component-wise-updates/
#
# https://pymc-devs.github.io/pymc/theory.html

"""
Take a sample from a distribution, specified by the inverse of its
cdf using inverse transform sampling.

Input:
- invcdf -- inverse of the cumulative density function

Output
- one sample

Notes:

- only works for 1D
- ApproxFun.jl can do this more generically.
"""
inverse_transform_sample(invcdf) = invcdf(rand())

"""
Take a sample from a distribution, specified by a pdf, over a given
range using rejection sampling.  Uses a uniform distribution as
proposal distribution.

Input:

- pdf -- the pdf to sample (needn't be normalized)
- domain -- the range over which to take samples (two vectors if multivariate)
- max_density -- the maximal value of the distribution (or an upper bound estimate)
- ntries=10^7 -- max number of Monte Carlo steps

https://en.wikipedia.org/wiki/Rejection_sampling

Note, ApproxFun can be used for 1D sampling too and may be much faster.
"""
function rejection_sample_unif(pdf, domain, max_density, ntries=10^7)
    dr = (domain[2]-domain[1])
    ndims = size(dr)
    # use Uniform(range) as distribution g(x)
    @inbounds for i=1:ntries
        xt = dr.*rand_(ndims)+domain[1]
        if pdf(xt)/max_density >= rand()
            return xt
        end
    end
    error("No sample found.")
end
rand_(::Tuple{}) = rand()
rand_(i) = rand(i)

"""
Take a sample from a distribution using rejection sampling using an
arbitrary proposal distribution.

Input:
- pdf -- pdf of distribution to sample from (needn't be normalized)
- ppdf -- proposal distribution pdf.  The closer to `pdf` it is the more efficient it is.
  Note:

  - ppdf must be scaled such that ppdf(x)>=pdf(x) for all x in the range.
  - ppdf needn't be normalized.

- sample_ppdf -- draws a sample from ppdf
- ntries=10^7 -- max number of Monte Carlo steps

Output:
- a sample from pdfX

Note: this is not type stable in 0.4
"""
function rejection_sample(pdf, ppdf, sample_ppdf, ntries=10^7)
    @inbounds for i=1:ntries
        xt = sample_ppdf() # doing ::Float64 helps a tiny bit
        if pdf(xt)/ppdf(xt) > rand()
            return xt
        end
    end
    error("No sample found.")
end


## MCMC samplers
################

## First a few helpers to write generic code:
#######

"Types used for dispatch depending on whether using a log-pdf or not."
@compat abstract type PDFType end
immutable LogPDF<:PDFType end
comp2zero_nan(p, ::LogPDF) = p==-Inf || isnan(p)
ratio(p1,p0, ::LogPDF) = p1-p0
multiply(p1,p0, ::LogPDF) = p1+p0
delog(p1, ::LogPDF) = exp(p1)

immutable NonLogPDF<:PDFType end
comp2zero_nan(p, ::NonLogPDF) = p==0 || isnan(p)
ratio(p1,p0, ::NonLogPDF) = p1/p0
multiply(p1,p0, ::NonLogPDF) = p1*p0
delog(p1, ::NonLogPDF) = p1

"""
Create (output) storage array or vector, to be used with _setindex!
`_make_storage(theta0, ni, nj)`

 - if typeof(theta0)==Vector then Array{eltype(theta0)}(length(theta0), niter-nburnin)
 - if typeof(theta0)!=Vector then Array{typeof(theta0)}(niter-nburnin)
"""
function _make_storage end
# Create no storage for nothings
_make_storage(theta0::Void, args...) = nothing
# Storing niter-nburnin values in a Vector or Array
_make_storage(theta::Vector, niter, nburnin) = Array{eltype(theta)}(length(theta), niter-nburnin)
_make_storage(theta, niter, nburnin) = Array{eltype(theta)}(niter-nburnin)
# Storing (niter-nburnin)/nthin values in a Vector or Array
_make_storage(theta0::Vector, niter, nburnin, nthin) = Array{eltype(theta0)}(length(theta0), (niter-nburnin)÷nthin)
_make_storage(theta0, niter, nburnin, nthin) = Array{eltype(theta0)}((niter-nburnin)÷nthin)
# Storing (niter-nburnin)/nthin x nchain values in 2D or 3D array
_make_storage(theta0::Vector, niter, nburnin, nthin, nchains) = Array{eltype(theta0)}(length(theta0), (niter-nburnin)÷nthin, nchains)
_make_storage(theta0, niter, nburnin, nthin, nchains) = Array{eltype(theta0)}((niter-nburnin)÷nthin, nchains)


"""
Used to be able to write code which is agnostic to vector-like or scalar-like parameters

`_setindex!(samples, theta, n[, nc])`

- if typeof(theta)==Vector then writes to a Matrix
- if typeof(theta)!=Vector then writes to a Vector{Vector}

Note that internally a copy (but not deepcopy) is made of the input.
"""
function _setindex! end
# fallback for blobs of ::Void
_setindex!(::Void, args...) = nothing
_setindex!(samples::AbstractArray, theta::Vector, i, nc=1) = samples[:,i,nc]=theta
_setindex!(samples::AbstractArray, theta, i, nc=1) = samples[i, nc]=copy(theta)


"""
Initializes storage and other bits for MCMCs.  Quite complicated to
allow for all the different input combinations.

If nchains==0 then assume that this MCMC cannot handle several chains.
"""
function _initialize(pdf_, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob, blob_reduce!; make_SharedArray=false)

    ntries = max(100,nburnin÷100) # how many tries per chain to find a IC with nonzero density

    pdftype = logpdf ? LogPDF() : NonLogPDF()
    pdf = hasblob ? pdf_ : theta -> (pdf_(theta), nothing)

    # Make initial chain positions and value of blob0 and p0
    if nchains==0 # a single theta is given and no chains are used, i.e. a single chain algo
        p0, blob0 = pdf(theta0)
        comp2zero_nan(p0, pdftype) && error("theta0=$(theta0) has zero density.  Choose another theta0.")
    elseif isa(theta0, Tuple) # Case: (theta0, ball_radius) Can only be used with Float parameters
        comp2zero_nan(pdf(theta0[1])[1], pdftype) && error("theta0[1]=$(theta0[1]) has zero density.  Choose another theta0[1].")

        # Initialize loop storage
        T = typeof(theta0[1])
        p0s = Float64[]
        sizehint!(p0s, nchains)
        theta0s = T[]
        sizehint!(theta0s, nchains)
        p0, blob0 = pdf(theta0[1])
        blob0s = typeof(blob0)[]
        sizehint!(blob0s, nchains)


        if isa(theta0[1], Number)
            radius = theta0[2]
            for i=1:nchains
                j = 0
                for j=1:ntries
                    tmp = theta0[1] + rand()*radius
                    p0, blob0 = pdf(tmp)
                    if !comp2zero_nan(p0,pdftype)
                        push!(theta0s, tmp)
                        push!(blob0s, blob0)
                        push!(p0s, p0)
                        break
                    end
                end
                j==ntries && error("Could not find suitable initial theta.  PDF is zero in too many places inside ball.")
            end
        else # assume theta0[1] is a Vector of numbers
            npara = length(theta0[1])
            radius = isa(theta0[2], Number) ? theta0[2]*ones(npara) : theta0[2]
            @assert length(radius)==npara
            for i=1:nchains
                j = 0
                for j=1:ntries
                    tmp = theta0[1] + randn(npara).*radius
                    p0, blob0 = pdf(tmp)
                    if !comp2zero_nan(p0,pdftype)
                        push!(theta0s, tmp)
                        push!(blob0s, blob0)
                        push!(p0s, p0)
                        break
                    end
                end
                j==ntries && error("Could not find suitable initial theta.  PDF is zero in too many places inside ball.")
            end
        end
    elseif isa(theta0, AbstractVector) # Case: all theta0 are given
        nchains = length(theta0)

        # Initialize loop storage
        T = typeof(theta0[1])
        theta0s = convert(Vector{T}, theta0)
        p0s = Float64[]
        sizehint!(p0s, nchains)
        p0, blob0 = pdf(theta0s[1])
        blob0s = typeof(blob0)[]
        sizehint!(blob0s, nchains)

        if !isleaftype(eltype(theta0s))
            warn("The input theta0 Vector has abstract eltype: $(eltype(theta0s)).  This will hurt performance!")
        end
        # Check that IC has nonzero pdf, otherwise drop it:
        chains2drop = Int[]
        for nc =1:nchains
            p0, blob0 = pdf(theta0s[nc])
            if comp2zero_nan(p0,pdftype) #p0==0
                warning("""Initial parameters of chain #$nc have zero probability density.
                        Skipping this chain.
                        theta0=$(theta0s[nc])
                        """)
                push!(chains2drop, nc)
            else
                # initialize p0s and blob0s
                push!(p0s, p0)
                push!(blob0s, blob0)
            end
        end
        deleteat!(theta0s, chains2drop)
        deleteat!(p0s, chains2drop)
        deleteat!(blob0s, chains2drop)
        nchains = length(theta0s)
    else
        error("Bad theta0, check docs for allowed theta zero input.")
    end


    if nchains==0
        # Initialize output storage
        thetas = _make_storage(theta0, niter, nburnin, nthin)
        logposts = _make_storage(1.0, niter, nburnin, nthin)
        blobs = blob_reduce!(blob0, niter, nburnin, nthin)
        return pdf, p0, theta0, blob0, thetas, blobs, nchains, pdftype, logposts
    else
        # Initialize output storage
        thetas = _make_storage(theta0s[1], niter, nburnin, nthin, nchains)
        logposts = _make_storage(1.0, niter, nburnin, nthin, nchains)
        blobs = blob_reduce!(blob0s[1], niter, nburnin, nthin, nchains)

        if make_SharedArray
            if !isbits(eltype(theta0s[1]))
                # SharedArrays only work with isbits types
                error("Only isbits types supported in parallel runs for eltype(theta).")
            end
            p0s, thetas = map(x->convert(SharedArray,x), (p0s, thetas))
            # Each worker will have its own copy of theta0s.  Considering the size of thetas this is fine.
            # theta0s = distribute(theta0s)
            if blobs!=nothing
                if !isbits(eltype(blob0s[1]))
                    error("Only isbits types supported in parallel runs for eltype(blob).")
                end
                blobs = convert(SharedArray, blobs)
                # Each worker will have its own copy of blob0s.  Considering the size of blobs this is fine.
                # blob0s = distribute(blob0s)
            end

        end
        return pdf, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype, logposts
    end
end


# initialize storage:
default_blob_reduce!(new_blob, niter::Int, nburnin::Int, nthin::Int) =
    _make_storage(new_blob, niter, nburnin, nthin)
default_blob_reduce!(new_blob, niter::Int, nburnin::Int, nthin::Int, nchains::Int) =
    _make_storage(new_blob, niter, nburnin, nthin, nchains)
# update storage-blob
default_blob_reduce!(stored_blob, new_blob, ni::Int) = _setindex!(stored_blob, new_blob, ni)
default_blob_reduce!(stored_blob, new_blob, ni::Int, nc::Int) = _setindex!(stored_blob, new_blob, ni, nc)
# squash chains
function default_blob_reduce!{T}(blobs::Array{T,3}, chains2keep)
    t = blobs[:,:,chains2keep]
    reshape(t, (size(t,1), size(t,2)*size(t,3)) )
end
default_blob_reduce!{T}(blobs::Array{T,2}, chains2keep) = blobs[:,chains2keep][:]
default_blob_reduce!{T}(blobs::Array{T,1}, chains2keep) = blobs[chains2keep]

## The serial MCMC samplers
###########################

"""
Metropolis sampler, the simplest MCMC. Can only be used with symmetric
proposal distributions (`ppdf`).

Input:

- pdf -- The probability density function to sample, defaults to
         log-pdf.  (The likelihood*prior in a Bayesian setting) Returns
         the density.

         If hasblob==true, then it also returns an arbitrary blob of
         something.
         `p, blob = pdf(theta)` where `theta` are the parameters.
         Note that blob can use pre-allocated memory as it will be copied
         (but not deepcopied) before storage.

- sample_ppdf -- draws a sample for the proposal/jump distribution
                 `sample_ppdf(theta)`
- theta0 -- initial value of parameters (<:AbstractVector)

Optional keyword input:

- niter -- number of steps to take (10^5)
- nburnin -- number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true (default) (for log-likelihoods) or false
- hasblob -- set to true if pdf also returns a blob
- blob_reduce! -- a function which updates the stored-blob with a
                     new blob, eg. to accommodate calculations made
                     with OnlineStats.jl. See below section.
- use_progress_meter=true : whether to show a progress meter

Output:

- samples:
  - if typeof(theta0)==Vector then Array{eltype(theta0)}(length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array{typeof(theta0)}(niter-nburnin)
- blobs: anything else that the pdf-function returns as second argument
- accept_ratio: ratio accepted to total steps
- logposterior: the value of the log-posterior for each sample

## Blobs and reduction:

The reduce-function needs to have two methods: one to initialize the
storage blob `new_blob -> storage_blob` and one to reduce a new blob
into the storage blob `(stored_blob, new_blob, ni[, nc]) -> nothing`
(updating stored_blob in-place).  `ni` is the iteration number, `nc`
is the chain number, if applicable.



"""
function metropolis(pdf, sample_ppdf, theta0;
                    niter=10^5,
                    nburnin=niter÷2,
                    logpdf=true,
                    nthin=1,
                    hasblob=false,
                    blob_reduce! = default_blob_reduce!,
                    use_progress_meter=true,
                    )
    if !hasblob
        blob_reduce! = default_blob_reduce!
    end
    # initialize
    nchains = 0
    pdf_, p0, theta0, blob0, thetas, blobs, nchains, pdftype, logposts =
        _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob, blob_reduce!, make_SharedArray=false)

    prog = use_progress_meter ? Progress(length((1-nburnin):(niter-nburnin))÷nthin, 1, "Metropolis, niter=$niter: ", 25) : nothing

    # run
    _metropolis!(p0, theta0, blob0, thetas, blobs, pdf_, sample_ppdf, niter, nburnin, nthin, pdftype, blob_reduce!, prog, logposts)
end
function _metropolis!(p0, theta0, blob0, thetas, blobs, pdf, sample_ppdf, niter, nburnin, nthin, pdftype, blob_reduce!, prog, logposts)
    naccept = 0
    ni = 1
    rng = (1-nburnin):(niter-nburnin)
    nn = 1
    @inbounds for n=rng
        # take a step:
        theta1 = sample_ppdf(theta0)
        p1, blob1 = pdf(theta1)
        # if p1/p0>rand() then accept:
        if  delog(ratio(p1,p0, pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
            theta0 = theta1
            blob0 = blob1
            p0 = p1
            naccept += 1
        end
        if rem(n,nthin)==0
            prog!=nothing && ProgressMeter.next!(prog; showvalues = [(:accept_ratio, signif(naccept/nn,3)), (:burnin_phase, n<=0)])
            if n>0
                _setindex!(thetas, theta0, ni)
                blob_reduce!(blobs, blob0, ni)
                logposts[ni] = p0
                ni +=1
            end
        end
        nn +=1 # number of steps
        if n==0 # reset for non-burnin phase
            naccept=0
            nn=1
        end
    end
    accept_ratio = naccept/(niter-nburnin)
    return thetas, accept_ratio, blobs, logposts
end

## TODO:
# - Metropolis-Hastings
# - block-wise

####
# The MC hammer: emcee implementation
###
# https://github.com/dfm/emcee

"""
The emcee MCMC sampler: https://github.com/dfm/emcee

Input:

- pdf -- The probability density function to sample, defaults to
         log-pdf.  The likelihood*prior in a Bayesian setting.
         `pdf` returns the density.

         If hasblob==true, then it also returns an arbitrary blob of
         something.
         `p, blob = pdf(theta)` where `theta` are the parameters.
         Note that blob can use pre-allocated memory as it will be copied
         (but not deepcopied) before storage.

- theta0 -- initial parameters.
  - If a tuple: choose random thetas around theta0[1] (<:AbstractVector) in a ball of radius theta0[2]
  - If a vector of vectors: use as starting points for an equal number of chains.

Optional key-word input:

- nchain -- number of chain to use (10^3).  If theta0 is vector of vectors, then set accordingly.
- niter -- total number of steps to take (10^5) (==total number of posterior evaluations).
- nburnin -- total number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true  (default) (for log-likelihoods) or false
- hasblob -- set to true if pdf also returns a blob
- use_progress_meter=true : whether to show a progress meter

Output:

- samples:
  - if typeof(theta0)==Vector then Array{eltype(theta0)}(length(theta0), niter-nburnin, nchains)
  - if typeof(theta0)!=Vector then Array{typeof(theta0)}(niter-nburnin, nchains)
- blobs: anything else that the pdf-function returns as second argument
- accept_ratio: ratio accepted to total steps
- logposterior: the value of the log-posterior for each sample

Note: use `squash_chains` to concatenate all chains into one chain.

Reference: emcee: The MCMC hammer, Foreman-Mackey et al. 2013

Example

    thetas, _, accept_ratio, logposteriors = emcee(x->sum(-x.^2), ([1.0,2, 5], 0.1))
"""
function emcee(pdf, theta0;
               nchains=10^2,
               niter=10^5,
               nburnin=niter÷2,
               logpdf=true,
               nthin=1,
               a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
               hasblob=false,
               blob_reduce! =default_blob_reduce!, # note the space after `!`
               use_progress_meter=true
               )
    @assert a_scale>1
    if !hasblob
        blob_reduce! = default_blob_reduce!
    end
    niter_emcee = niter ÷ nchains
    nburnin_emcee = nburnin ÷ nchains

    # initialize
    pdf_, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype, logposts =
        _initialize(pdf, theta0, niter_emcee, nburnin_emcee, logpdf, nchains, nthin, hasblob, blob_reduce!, make_SharedArray=false)
    # initialize progress meter (type-unstable)
    prog = use_progress_meter ? Progress(length((1-nburnin_emcee):(niter_emcee-nburnin_emcee)), 1, "emcee, niter=$niter, nchains=$nchains: ", 25) : nothing
    # do the MCMC
    _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf_, niter_emcee, nburnin_emcee, nchains, nthin, pdftype, a_scale, blob_reduce!, prog, logposts)
end

function _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter_emcee, nburnin_emcee, nchains, nthin, pdftype, a_scale, blob_reduce!, prog, logposts)
    # initialization and work arrays:
    naccept = zeros(Int, nchains)
    ni = ones(Int, nchains)
    N = length(theta0s[1])
    nn = 1
    rng = (1-nburnin_emcee):(niter_emcee-nburnin_emcee)
    @inbounds for n = rng
        for nc = 1:nchains
            # draw a random other chain
            no = rand(1:nchains-1)
            no = no>=nc ? no+1 : no # shift by one
            # sample g (eq. 10)
            z = sample_g(a_scale)

            # sample a new theta with the stretch move:
            theta1 = theta0s[no] + z*(theta0s[nc]-theta0s[no]) # eq. 7
            # and calculate the theta1 density:
            p1, blob1 = pdf(theta1)

            # if z^(N-1)*p1/p0>rand() then accept:
            if z^(N-1)*delog(ratio(p1,p0s[nc], pdftype), pdftype)>=rand() # ugly because of log & non-log pdfs
                theta0s[nc] = theta1
                p0s[nc] = p1

                blob0s[nc] = blob1
                naccept[nc] += 1
            end
            # store theta after burn-in
            if  n>0 && rem(n,nthin)==0
                _setindex!(thetas, theta0s[nc], ni[nc], nc)
                blob_reduce!(blobs, blob0s[nc], ni[nc], nc)
                logposts[ni[nc], nc] = p0s[nc]
                ni[nc] +=1
            end
        end # for nc =1:nchains
        if n==0
            naccept = zeros(naccept)
            nn=1
        end
        macc = mean(naccept)
        sacc = sqrt(var(naccept)) # std(naccept) is not type-stable!
        outl = sum(abs.(naccept.-macc).>2*sacc)
        prog!=nothing && ProgressMeter.next!(prog;
                                             showvalues = [(:accept_ratio_mean, signif(macc/nn,3)),
                                                           (:accept_ratio_std, signif(sacc/nn,3)),
                                                           (:accept_ratio_outliers, outl),
                                                           (:burnin_phase, n<=0)])
        nn +=1
    end # for n=(1-nburnin_emcee):(niter_emcee-nburnin_emcee)

    accept_ratio = [na/(niter_emcee-nburnin_emcee) for na in naccept]
    return thetas, accept_ratio, blobs, logposts
end

"g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
g_pdf(z, a) = 1/a<=z<=a ? 1/sqrt(z) * 1/(2*(sqrt(a)-sqrt(1/a))) : zero(z)

"Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
cdf_g_inv(u, a) = (u*(sqrt(a)-sqrt(1/a)) + sqrt(1/a) )^2

"Sample from g using inverse transform sampling.  a=2.0 is recommended."
sample_g(a) = cdf_g_inv(rand(), a)

"""
    evaluate_convergence(thetas; indices=1:size(thetas)[1])

Evaluates:
- R̂ (potential scale reduction): should be <1.05
- total effective sample size of all chains combined, and
- the average thinning factor

See Gelman etal 2014, page 281-287

Example

    thetas, _, accept_ratio, logposteriors = emcee(x->sum(-x.^2), ([1.0,2, 5], 0.1))
    Rhat, sample_size, nthin = evaluate_convergence(thetas)
"""
function evaluate_convergence(thetas; indices=1:size(thetas)[1])
    Rs = Float64[]
    sample_size = Float64[]
    nchains = size(thetas)[3]
    split = size(thetas)[2]÷2
    for i in indices
        chains = vcat([thetas[i,1:split,n] for n=1:nchains], [thetas[i,split+1:2*split,n] for n=1:nchains])
        push!(Rs, potential_scale_reduction(chains...))
        push!(sample_size, sum(effective_sample_size.(chains)))
    end
    nthin = size(thetas)[2]*size(thetas)[3]/mean(sample_size)
    return Rs, sample_size, isnan(nthin) ? -1 : round(Int, nthin)
end



"""
    squash_chains(thetas, accept_ratio=zeros(size(thetas)[end]), blobs=nothing;
                                                                 drop_low_accept_ratio=false,
                                                                 drop_fact=1,
                                                                 blob_reduce! =default_blob_reduce!,
                                                                 verbose=true,
                                                                 order=false
                                                                 )
Puts the samples of all chains into one vector.  Note that afterwards Rhat cannot
be computed anymore.

This can also drop chains which have a low accept ratio (this can happen with
emcee), by setting drop_low_accept_ratio (Although whether it is wise to
do so is another question).  drop_fact -> increase to drop more chains.

Returns:
- theta
- mean(accept_ratio[chains2keep])
- blobs
- log-posteriors
- reshape_revert: reshape(thetas_out, reshape_revert...) will put it back into the chains.
"""
function squash_chains(thetas, accept_ratio=zeros(size(thetas)[end]), blobs=nothing, logposts=nothing; drop_low_accept_ratio=false,
                                                                 drop_fact=2,
                                                                 blob_reduce! =default_blob_reduce!,
                                                                 verbose=true,
                                                                 order=false
                                                                 )
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
        chains2keep = collect(1:nchains) # collect to make chains2keep::Vector{Int}
    end

    # TODO: below creates too many temporary arrays
    if ndims(thetas)==3
        t = thetas[:,:,chains2keep]
        reshape_revert = size(t)
        t = reshape(t, (size(t,1), size(t,2)*size(t,3)) )
    else
        t = thetas[:,chains2keep]
        reshape_revert = size(t)
        t = t[:]
    end
    if logposts==nothing
        l = nothing
    else
        l = logposts[:,chains2keep][:]
    end
    if blobs==nothing
        b = nothing
    else
        b = blob_reduce!(blobs, chains2keep)
    end
    if order # order such that early samples are early in thetas
        nc = length(chains2keep)
        ns = size(thetas,1)
        perm = sortperm(vcat([collect(1:ns) for i=1:nc]...))
        if blobs!=nothing
            b = b[perm]
        end
        if logposts!=nothing
            l = l[perm]
        end
        if ndims(thetas)==3
            t = t[:,perm]
        else
            t = t[perm]
        end
    end
    return t, mean(accept_ratio[chains2keep]), b, l, reshape_revert
end
