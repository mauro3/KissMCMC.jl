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
abstract PDFType
immutable LogPDF<:PDFType end
comp2zero(p, ::LogPDF) = p==-Inf
ratio(p1,p0, ::LogPDF) = p1-p0
multiply(p1,p0, ::LogPDF) = p1+p0
delog(p1, ::LogPDF) = exp(p1)

immutable NonLogPDF<:PDFType end
comp2zero(p, ::NonLogPDF) = p==0
ratio(p1,p0, ::NonLogPDF) = p1/p0
multiply(p1,p0, ::NonLogPDF) = p1*p0
delog(p1, ::NonLogPDF) = p1

"""
Create (output) storage array or vector, to be used with _setindex!
`_make_storage(theta0, ni, nj)`

 - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
 - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
"""
function _make_storage end
# Create no storage for nothings
_make_storage(theta0::Void, args...) = nothing
# Storing niter-nburnin values in a Vector or Array
_make_storage(theta::Vector, niter, nburnin) = Array(eltype(theta), length(theta), niter-nburnin)
_make_storage(theta, niter, nburnin) = Array(eltype(theta), niter-nburnin)
# Storing (niter-nburnin)/nthin values in a Vector or Array
_make_storage(theta0::Vector, niter, nburnin, nthin) = Array(eltype(theta0), length(theta0), floor(Int, (niter-nburnin)/nthin))
_make_storage(theta0, niter, nburnin, nthin) = Array(eltype(theta0), floor(Int, (niter-nburnin)/nthin))
# Storing (niter-nburnin)/nthin x nchain values in 2D or 3D array
_make_storage(theta0::Vector, niter, nburnin, nthin, nchains) = Array(eltype(theta0), length(theta0), nchains, floor(Int, (niter-nburnin)/nthin))
_make_storage(theta0, niter, nburnin, nthin, nchains) = Array(eltype(theta0), nchains, floor(Int, (niter-nburnin)/nthin))


"""
Used to be able to write code which is agnostic to vector-like or scalar-like parameters

`_setindex!(samples, theta, n)`

 - if typeof(samples)!=Vector then writes to a Matrix
 - if typeof(samples)==Vector then writes to a Vector{Vector}

Note that internally a copy (but not deepcopy) is made of the input.
"""
function _setindex! end
# fallback for blobs of ::Void
_setindex!(::Void, args...) = nothing
# Matrix{Scalar} vs Vector{Vector}
_setindex!(samples::AbstractMatrix, theta, n) = samples[:,n]=theta
_setindex!(samples, theta, n) = samples[n]=copy(theta)
#
_setindex!{T}(samples::AbstractArray{T,3}, theta, nw, i) = samples[:,nw,i]=theta
_setindex!{T}(samples::AbstractArray{T,2}, theta, nw, i) = samples[nw, i]=copy(theta)


"""
Initializes storage and other bits for MCMCs.  Quite complicated to
allow for all the different input combinations.

If nchains==0 then assume that this MCMC cannot handle several chains.
"""
function _initialize(pdf_, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob; make_SharedArray=false)

    ntries = 100 # how many tries to find a IC with nonzero density

    pdftype = logpdf ? LogPDF() : NonLogPDF()
    pdf = hasblob ? pdf_ : theta -> (pdf_(theta), nothing)

    # Make initial chain positions and value of blob0 and p0
    if nchains==0 # a single theta is given an no chains are used, i.e. a single chain algo
        p0, blob0 = pdf(theta0)
        comp2zero(p0, pdftype) && error("theta0=$(theta0) has zero density.  Choose another theta0.")
    elseif isa(theta0, Tuple) # Case: (theta0, ball_radius) Can only be used with Float parameters
        comp2zero(pdf(theta0[1])[1], pdftype) && error("theta0[1]=$(theta0[1]) has zero density.  Choose another theta0[1].")

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
                    if !comp2zero(p0,pdftype)
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
                    tmp = theta0[1] + rand(npara).*radius
                    p0, blob0 = pdf(tmp)
                    if !comp2zero(p0,pdftype)
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
        for nw =1:nchains
            p0, blob0 = pdf(theta0s[nw])
            if comp2zero(p0,pdftype) #p0==0
                warning("""Initial parameters of chain #$nw have zero probability density.
                        Skipping this chain.
                        theta0=$(theta0s[nw])
                        """)
                push!(chains2drop, nw)
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
        blobs = _make_storage(blob0, niter, nburnin, nthin)
        return pdf, p0, theta0, blob0, thetas, blobs, nchains, pdftype
    else
        # Initialize output storage
        thetas = _make_storage(theta0s[1], niter, nburnin, nthin, nchains)
        blobs = _make_storage(blob0s[1], niter, nburnin, nthin, nchains)

        if make_SharedArray
            if !isbits(eltype(theta0s[1]))
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
        return pdf, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype
    end
end


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
- theta0 -- initial value of parameters

Optional keyword input:

- niter -- number of steps to take (10^5)
- nburnin -- number of initial steps discarded, aka burn-in (niter/10)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true (default) (for log-likelihoods) or false
- hasblob -- set to true if pdf also returns a blob

Output:

- samples:
  - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
- blobs: anything else that the pdf-function returns as second argument
- accept_ratio: ratio accepted to total steps

"""
function metropolis(pdf, sample_ppdf, theta0;
                    niter=10^5,
                    nburnin=niter÷10,
                    logpdf=true,
                    nthin=1,
                    hasblob=false
                    )

    # intialize
    nchains = 0
    pdf_, p0, theta0, blob0, thetas, blobs, nchains, pdftype =
        _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob, make_SharedArray=true)

    # run
    _metropolis!(p0, theta0, blob0, thetas, blobs, pdf_, sample_ppdf, niter, nburnin, nthin, pdftype)
end
function _metropolis!(p0, theta0, blob0, thetas, blobs, pdf, sample_ppdf, niter, nburnin, nthin, pdftype)
    naccept = 0
    ni = 1
    @inbounds for n=(1-nburnin):(niter-nburnin)
        # take a step:
        theta1 = sample_ppdf(theta0)
        p1, blob1 = pdf(theta1)
        # if p1/p0>rand() then accept:
        if  delog(ratio(p1,p0, pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
            theta0 = theta1
            blob0 = blob1
            p0 = p1
            if n>0
                naccept += 1
            end
        end
        if n>0 && rem(n,nthin)==0
            _setindex!(thetas, theta0, ni)
            _setindex!(blobs, blob0, ni)
            ni +=1
        end
    end
    accept_ratio = naccept/length(thetas)
    return thetas, accept_ratio, blobs
end

## TODO:
# - Metropolis-Hastings
# - block-wise

####
# The MC hammer: emcee implementation
###
# https://github.com/dfm/emcee

"""
The emcee Metropolis sampler: https://github.com/dfm/emcee

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
  - If a tuple: choose random thetas around theta0[1] in a ball of radius theta0[2]
  - If a vector of vectors: use given starting points and that number of chains.

Optional key-word input:

- nchain -- number of chain to use (10^3).  If theta0 is vector of vectors, then set accordingly.
- niter -- number of steps to take (10^4)
           Thus the total number of likelihood! evaluations is nchain*niter
- nburnin -- number of initial steps discarded, aka burn-in (niter/10)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true  (default) (for log-likelihoods) or false
- hasblob -- set to true if pdf also returns a blob

Output:

- samples:
  - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
- blobs: anything else that the pdf-function returns as second argument
- accept_ratio: ratio accepted to total steps

Note: use `squash_chains` to concatenate all chains into one chain.

Reference: emcee: The MCMC hammer, Foreman-Mackey et al. 2013
"""
function emcee(pdf, theta0;
               nchains=10^2,
               niter=10^4,
               nburnin=niter÷10,
               logpdf=true,
               nthin=1,
               a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
               hasblob=false
               )
    # initialize
    pdf_, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype =
        _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob, make_SharedArray=false)
    # do the MCMC
    _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf_, niter, nburnin, nchains, nthin, pdftype, a_scale)
end
function _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter, nburnin, nchains, nthin, pdftype, a_scale)
    # initialization and work arrays:
    naccept = zeros(Int, nchains)
    ni = ones(Int, nchains)
    N = length(theta0s[1])

    @inbounds for n = (1-nburnin):(niter-nburnin)
        for nw = 1:nchains
            # draw a random other chain
            no = rand(1:nchains-1)
            no = no>=nw ? no+1 : no # shift by one
            # sample g (eq. 10)
            z = sample_g(a_scale)

            # propose new step with the stretch move:
            theta1 = theta0s[no] + z*(theta0s[nw]-theta0s[no]) # eq. 7
            # and its pdf:
            p1, blob1 = pdf(theta1)

            # if z^(N-1)*p1/p0>rand() then accept:
            if z^(N-1)*delog(ratio(p1,p0s[nw], pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
                theta0s[nw] = theta1
                p0s[nw] = p1

                blob0s[nw] = blob1
                if n>0
                    naccept[nw] += 1
                end
            end
            if  n>0 && rem(n,nthin)==0
                _setindex!(thetas, theta0s[nw], nw, ni[nw])
                _setindex!(blobs, blob0s[nw], nw, ni[nw])
                ni[nw] +=1
            end
        end # for nw =1:nchains
    end # for n=(1-nburnin):(niter-nburnin)

    accept_ratio = [na/(niter-nburnin) for na in naccept]
    return thetas, accept_ratio, blobs
end

"Puts the samples of all chains into one vector."
function squash_chains(thetas, accept_ratio=-1.0, blobs=nothing)
    if ndims(thetas)==3
        return thetas[:,:], mean(accept_ratio), blobs==nothing ? nothing : blobs[:,:]
    else
        return thetas[:], mean(accept_ratio), blobs==nothing ? nothing : blobs[:]
    end
end

"g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
g_pdf(z, a=2.0) = 1/a<=z<=a ? 1/sqrt(z) * 1/(2*(sqrt(a)-sqrt(1/a))) : zero(z)

"""
Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013.
"""
cdf_g_inv(u, a=2.0) = (u*(sqrt(a)-sqrt(1/a)) + sqrt(1/a) )^2

"Sample from g using inverse transform sampling"
sample_g(a=2.0) = cdf_g_inv(rand(), a)
