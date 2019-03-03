# A bunch of parallel MCMC probability density function samplers.
#
# Refs:
# https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
# https://theclevermachine.wordpress.com/2012/11/04/mcmc-multivariate-distributions-block-wise-component-wise-updates/
#
# https://pymc-devs.github.io/pymc/theory.html

## The parallel MCMC samplers
###########################

getlocalindex(rng, nc) = findfirst(rng,nc)

"""
Parallel Metropolis sampler, the simplest MCMC. Can only be used with symmetric
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
- nburnin -- number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true (default) (for log-likelihoods) or false

Output:

- samples:
  - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
- accept_ratio: ratio accepted to total steps
- blobs: anything else that the pdf-function returns as second argument
- logposterior: the value of the log-posterior for each sample
"""
function metropolisp(pdf, sample_ppdf, theta0;
                     niter=10^5,
                     nburnin=niter÷2,
                     nchains=nothing,
                     nthin=1,
                     logpdf=true,
                     hasblob=false,
                     blob_reduce! = default_blob_reduce!,
                     )
    if nchains!=nothing
        warn("nchains keyword not supported")
    end
    # intialize
    pdf_, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype, logposts =
        _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, hasblob, blob_reduce!, make_SharedArray=true)
    naccept = SharedArray{Int}(nchains, init = S->S[:]=0)
    ni = SharedArray{Int}(nchains, init = S->S[:]=1)

    # run
    _metropolisp!(p0s, theta0s, blob0s, thetas, blobs, pdf_, sample_ppdf, niter, nburnin, nchains, nthin, pdftype,
                  naccept, ni, blob_reduce!, logposts)
end
function _metropolisp!(p0s, theta0s, blob0s, thetas, blobs, pdf, sample_ppdf, niter, nburnin, nchains, nthin, pdftype,
                       naccept, ni, blob_reduce!, logposts)
    N = length(theta0s[1])
    @sync @parallel for nc=1:nchains
        @inbounds for n=(1-nburnin):(niter-nburnin)
            # take a step:
            theta1 = sample_ppdf(theta0s[nc])
            p1, blob1 = pdf(theta1)
            # if p1/p0>rand() then accept:
            if  delog(ratio(p1,p0s[nc], pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
                theta0s[nc] = theta1
                blob0s[nc] = blob1
                p0s[nc] = p1
                if n>0
                    naccept[nc] += 1
                end
            end
            if n>0 && rem(n,nthin)==0
                _setindex!(thetas, theta0s[nc], ni[nc], nc)
                blob_reduce!(blobs, blob0s[nc], ni[nc], nc)
                logposts[ni] = p0s[nc]
                ni[nc] +=1
            end
        end
    end
    accept_ratio = [na/(niter-nburnin) for na in naccept]
    return sdata(thetas), accept_ratio, blobs==nothing ? nothing : sdata(blobs), logposts
end

immutable IsScalar end
immutable IsVector end

# pre re-write
# emceep          :   4.776336 seconds (18.61 M allocations: 1.204 GB, 2.42% gc time)
# pre IsScalar:
# emceep          :   6.769637 seconds (66.36 M allocations: 2.032 GB, 2.90% gc time)
# after
# emceep          :   5.075266 seconds (17.16 M allocations: 1.277 GB, 3.94% gc time)
"""
Parallel emcee Metropolis sampler: https://github.com/dfm/emcee

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
- niter -- total number of steps to take (10^5) (==total number of posterior evaluations).
- nburnin -- total number of initial steps discarded, aka burn-in (niter/3)
- nthin -- only store every n-th sample (default=1)
- logpdf -- either true  (default) (for log-likelihoods) or false
- hasblob -- set to true if pdf also returns a blob
- ball_radius_halfing_steps=7 : if no initial theta can be round within the ball, its radius will be halved
                                and tried again; repeatedly for the specified amount of halfing-steps.

Output:

- samples:
  - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
  - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
- accept_ratio: ratio accepted to total steps
- blobs: anything else that the pdf-function returns as second argument
- logposterior: the value of the log-posterior for each sample

Note: use `squash_chains` to concatenate all chains into one chain.

Reference: emcee: The MCMC hammer, Foreman-Mackey et al. 2013
"""
function emceep(pdf, theta0;
                nchains=10^2,
                niter=10^5,
                nburnin=niter÷2,
                logpdf=true,
                nthin=1,
                a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
                hasblob=false,
                blob_reduce! = default_blob_reduce!,
                ball_radius_halfing_steps=7
                )
    niter_emcee = niter ÷ nchains
    nburnin_emcee = nburnin ÷ nchains

    pdf_, p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype, logposts =
        _initialize(pdf, theta0, niter_emcee, nburnin_emcee, logpdf, nchains, nthin, hasblob, blob_reduce!;
                    make_SharedArray=true, ball_radius_halfing_steps=ball_radius_halfing_steps)
    @assert nchains>=2*(length(theta0s[1])+2) "Use more chains: at least 2*(DOF+2), but better many more!"

    # make theta0s blob0s into SharedArray
    if eltype(theta0s)<:Number
        theta0s = convert(SharedArray,theta0s')
        isscalar = IsScalar()
    else
        # TODO: make efficient
        theta0s = hcat(theta0s...)
        theta0s = convert(SharedArray,theta0s)
        isscalar = IsVector()
    end
    if eltype(blob0s)<:Number
        blob0s = convert(SharedArray,blob0s)
    else
        # TODO: make efficient
        blob0s = hcat(blob0s...)
        if eltype(blob0s)!=Void
            blob0s = convert(SharedArray,blob0s)
        end
    end
    theta1 = isscalar==IsScalar() ? zero(eltype(theta0s)) : zeros(eltype(theta0s), size(theta0s,1))

    # initialization and work arrays:
    naccept = SharedArray{Int}(nchains, init = S->S[:]=0)
    ni = SharedArray{Int}(nchains, init = S->S[:]=1)
    # do the MCMC
    _parallel_emcee!(p0s, theta0s, blob0s, theta1, isscalar, thetas, blobs, pdf_,
                     niter_emcee, nburnin_emcee, nchains, nthin, pdftype, a_scale,
                     naccept, ni, blob_reduce!, logposts)
end

function _parallel_emcee!(p0s, theta0s, blob0s, theta1, isscalar, thetas, blobs, pdf,
                          niter_emcee, nburnin_emcee, nchains, nthin, pdftype, a_scale,
                          naccept, ni, blob_reduce!, logposts)
    N = size(theta0s,1) # number of parameters
    # the two sets
    nchains12 = UnitRange{Int}[1:nchains÷2, nchains÷2+1:nchains]

    #     @inbounds
    for n = (1-nburnin_emcee):(niter_emcee-nburnin_emcee)
        for i=1:2
            ncs = nchains12[i] # chains to update
            ncso = nchains12[mod1(i+1,2)] # other chains, to stretch-move with
            @sync @parallel for nc in ncs
                # draw a random other chain
                nco = rand(ncso)
                # sample g (eq. 10)
                z = sample_g(a_scale)

                # propose new step with stretch-move:
                theta1 = stretch_move!(theta1, theta0s, nc, nco, z, isscalar)

                # and its density:
                p1, blob1 = pdf(theta1)

                # if z^(N-1)*p1/p0>rand() then accept:
                # (ugly because of log & non-log pdfs)
                if z^(N-1)*delog(ratio(p1,p0s[nc], pdftype), pdftype)>rand()
                    theta0s[:,nc] = theta1
                    p0s[nc] = p1
                    blob0s[:,nc] = blob1
                    if n>0
                        naccept[nc] += 1
                    end
                end
                if  n>0 && rem(n,nthin)==0
                    if isscalar==IsScalar()
                        _setindex!(thetas, theta0s[:,nc][1], ni[nc], nc)
                        blob_reduce!(blobs, blob0s[:,nc][1], ni[nc], nc)
                    else
                        _setindex!(thetas, theta0s[:,nc], ni[nc], nc)
                        blob_reduce!(blobs, blob0s[:,nc], ni[nc], nc)
                    end
                    logposts[ni[nc], nc] = p0s[nc]
                    ni[nc] +=1
                end
             end # for nc in ncs
        end # for i=1:2
    end # for n=(1-nburnin_emcee):(niter_emcee-nburnin_emcee)

    accept_ratio = [na/(niter_emcee-nburnin_emcee) for na in naccept]

    return sdata(thetas), accept_ratio, blobs==nothing ? nothing : sdata(blobs), logposts
end


function stretch_move!(theta1, theta0s, nc, nco, z, ::IsVector)
    N = size(theta0s,1) # number of parameters
    for j=1:N
        theta1[j] = theta0s[j,nco] + z*(theta0s[j,nc]-theta0s[j,nco]) # eq. 7
    end
    return theta1
end
function stretch_move!(theta1, theta0s, nc, nco, z, ::IsScalar)
    return theta0s[1,nco][1] + z*(theta0s[1,nc][1]-theta0s[1,nco][1]) # eq. 7
end
# ## MCMC samplers
# ################

# ## First a few helpers to write generic code:
# #######

# "Types used for dispatch depending on whether using a log-pdf or not."
# abstract PDFType
# immutable LogPDF<:PDFType end
# comp2zero(p, ::LogPDF) = p==-Inf
# ratio(p1,p0, ::LogPDF) = p1-p0
# multiply(p1,p0, ::LogPDF) = p1+p0
# delog(p1, ::LogPDF) = exp(p1)

# immutable NonLogPDF<:PDFType end
# comp2zero(p, ::NonLogPDF) = p==0
# ratio(p1,p0, ::NonLogPDF) = p1/p0
# multiply(p1,p0, ::NonLogPDF) = p1*p0
# delog(p1, ::NonLogPDF) = p1

# """
# Create (output) storage array or vector, to be used with _setindex!
# `_make_storage(theta0, ni, nj)`

#  - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
#  - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
# """
# function _make_storage end
# # Create no storage for nothings
# _make_storage(theta0::Void, args...) = nothing
# # Storing niter-nburnin values in a Vector or Array
# _make_storage(theta::Vector, niter, nburnin) = Array(eltype(theta), length(theta), niter-nburnin)
# _make_storage(theta, niter, nburnin) = Array(eltype(theta), niter-nburnin)
# # Storing (niter-nburnin)/nthin values in a Vector or Array
# _make_storage(theta0::Vector, niter, nburnin, nthin) = Array(eltype(theta0), length(theta0), floor(Int, (niter-nburnin)/nthin))
# _make_storage(theta0, niter, nburnin, nthin) = Array(eltype(theta0), floor(Int, (niter-nburnin)/nthin))
# # Storing (niter-nburnin)/nthin x nchain values in 2D or 3D array
# _make_storage(theta0::Vector, niter, nburnin, nthin, nchains) = Array(eltype(theta0), length(theta0), nchains, floor(Int, (niter-nburnin)/nthin))
# _make_storage(theta0, niter, nburnin, nthin, nchains) = Array(eltype(theta0), nchains, floor(Int, (niter-nburnin)/nthin))


# """
# Used to be able to write code which is agnostic to vector-like or scalar-like parameters

# `_setindex!(samples, theta, n)`

#  - if typeof(samples)!=Vector then writes to a Matrix
#  - if typeof(samples)==Vector then writes to a Vector{Vector}

# Note that internally a copy (but not deepcopy) is made of the input.
# """
# function _setindex! end
# # fallback for blobs of ::Void
# _setindex!(::Void, args...) = nothing
# # Matrix{Scalar} vs Vector{Vector}
# _setindex!(samples::SharedMatrix, theta, n) = samples[:,n]=theta
# _setindex!(samples, theta, n) = samples[n]=copy(theta)
# #
# _setindex!{T}(samples::AbstractArray{T,3}, theta, nc, i) = samples[:,nc,i]=theta
# _setindex!{T}(samples::AbstractArray{T,2}, theta, nc, i) = samples[nc, i]=copy(theta)


# ## The parallel samplers
# ########################

# """
# Metropolis sampler, the simplest MCMC. Can only be used with symmetric
# proposal distributions (`ppdf`).  Can run several, completely
# independent, chains in parallel.

# Input:

# - pdf -- The probability density function to sample, defaults to
#          log-pdf.  (The likelihood*prior in a Bayesian setting) Returns
#          the pdf and an arbitrary other thing which gets stored
#          (return `nothing` to store nothing).

#          `p, blob = pdf(theta)` where `theta` are the parameters.
#          Note that blob can use pre-allocated memory as it will be copied
#          (but not deepcopied) before storage.

# - sample_ppdf -- draws a sample for the proposal/jump distribution
#                  `sample_ppdf(theta)`
# - theta0 -- initial value of parameters. Pass in several for several chains.

# Optional keyword input:

# - niter -- number of steps to take (10^5)
# - nburnin -- number of initial steps discarded, aka burn-in (niter/10)
# - nthin -- only store every n-th sample (default=1)
# - logpdf -- either true (default) (for log-likelihoods) or false

# Output:

# - samples:
#   - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
#   - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
# - blobs: anything else that the pdf-function returns as second argument
# - accept_ratio: ratio accepted to total steps

# """
# function metropolisp(pdf, sample_ppdf, theta0;
#                     niter=10^5,
#                     nburnin=niter÷10,
#                     nchains=length(theta0),
#                     nthin=1,
#                     logpdf=true,
#                     )
#     # intialize
#     p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype =
#         _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, make_SharedArray=true)
#     # run
#     _metropolisp!(p0s, theta0s, blob0s, thetas, blobs, pdf, sample_ppdf, niter, nburnin, nchains, nthin, pdftype)
# end
# function _metropolisp!(p0s, theta0s, blob0s, thetas, blobs, pdf, sample_ppdf, niter, nburnin, nchains, nthin, pdftype)
#     naccept = SharedArray{Int}(nchains, init = S->S[:]=0)
#     ni = SharedArray{Int}(nchains, init = S->S[:]=1)
#     N = length(theta0s[1])

#     @sync @parallel for nc=1:nchains
#         @inbounds for n=(1-nburnin):(niter-nburnin)
#             # take a step:
#             theta1 = sample_ppdf(theta0s[nc])
#             p1, blob1 = pdf(theta1)
#             # if p1/p0>rand() then accept:
#             if  delog(ratio(p1,p0s[nc], pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
#                 theta0s[nc] = theta1
#                 blob0s[nc] = blob1
#                 p0s[nc] = p1
#                 if n>0
#                     naccept[nc] += 1
#                 end
#             end
#             if n>0 && rem(n,nthin)==0
#                 _setindex!(thetas, theta0s[nc], ni[nc], nc)
#                 _setindex!(blobs, blob0s[nc], ni[nc], nc)
#                 ni[nc] +=1
#             end
#         end
#     end
#     accept_ratio = [na/(niter-nburnin) for na in naccept]
#     return sdata(thetas), blobs==nothing ? nothing : sdata(blobs), accept_ratio
# end


# """
# Initializes storage and other bits for MCMCs.  Quite complicated to
# allow for all the different input combinations.

# If nchains==0 then assume that this MCMC cannot handle several chains.
# """
# function _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin; make_SharedArray=false)

#     ntries = 100 # how many tries to find a IC with nonzero density

#     pdftype = logpdf ? LogPDF() : NonLogPDF()

#     # Make initial chain positions and value of blob0 and p0
#     if isa(theta0, Tuple) # Case: (theta0, ball_radius) Can only be used with Float parameters
#         comp2zero(pdf(theta0[1])[1], pdftype) && error("theta0[1]=$(theta0[1]) has zero density.  Choose another theta0[1].")

#         # Initialize loop storage
#         T = typeof(theta0[1])
#         p0s = Float64[]
#         sizehint!(p0s, nchains)
#         theta0s = T[]
#         sizehint!(theta0s, nchains)
#         p0, blob0 = pdf(theta0[1])
#         blob0s = typeof(blob0)[]
#         sizehint!(blob0s, nchains)


#         if isa(theta0[1], Number)
#             radius = theta0[2]
#             for i=1:nchains
#                 j = 0
#                 for j=1:ntries
#                     tmp = theta0[1] + rand()*radius
#                     p0, blob0 = pdf(tmp)
#                     if !comp2zero(p0,pdftype)
#                         push!(theta0s, tmp)
#                         push!(blob0s, blob0)
#                         push!(p0s, p0)
#                         break
#                     end
#                 end
#                 j==ntries && error("Could not find suitable initial theta.  PDF is zero in too many places inside ball.")
#             end
#         else # assume theta0 is a Vector of numbers
#             npara = length(theta0[1])
#             radius = isa(theta0[2], Number) ? theta0[2]*ones(npara) : theta0[2]
#             @assert length(radius)==npara
#             for i=1:nchains
#                 j = 0
#                 for j=1:ntries
#                     tmp = T[theta0[1][j] + rand()*radius[j] for j=1:npara]
#                     p0, blob0 = pdf(tmp)
#                     if !comp2zero(p0,pdftype)
#                         push!(theta0s, tmp)
#                         push!(blob0s, blob0)
#                         push!(p0s, p0)
#                         break
#                     end
#                 end
#                 j==ntries && error("Could not find suitable initial theta.  PDF is zero in too many places inside ball.")
#             end
#         end
#     elseif nchains==0 # a single theta is given an no chains are used, i.e. a single chain algo
#         p0, blob0 = pdf(theta0)
#         comp2zero(p0, pdftype) && error("theta0=$(theta0) has zero density.  Choose another theta0.")
#     elseif isa(theta0, AbstractVector) # Case: all theta0 are given
#         nchains = length(theta0)

#         # Initialize loop storage
#         T = typeof(theta0[1])
#         theta0s = convert(Vector{T}, theta0)
#         p0s = Float64[]
#         sizehint!(p0s, nchains)
#         p0, blob0 = pdf(theta0s[1])
#         blob0s = typeof(blob0)[]
#         sizehint!(blob0s, nchains)

#         if !isleaftype(eltype(theta0s))
#             warn("The input theta0 Vector has abstract eltype: $(eltype(theta0s)).  This will hurt performance!")
#         end
#         # Check that IC has nonzero pdf, otherwise drop it:
#         chains2drop = Int[]
#         for nc =1:nchains
#             p0, blob0 = pdf(theta0s[nc])
#             if comp2zero(p0,pdftype) #p0==0
#                 warning("""Initial parameters of chain #$nc have zero probability density.
#                             Skipping this chain.
#                             theta0=$(theta0s[nc])
#                             """)
#                 push!(chains2drop, nc)
#             else
#                 # initialize p0s and blob0s
#                 push!(p0s, p0)
#                 push!(blob0s, blob0)
#             end
#         end
#         deleteat!(theta0s, chains2drop)
#         deleteat!(p0s, chains2drop)
#         deleteat!(blob0s, chains2drop)
#         nchains = length(theta0s)
#     else
#         error("Bad theta0, check docs for allowed theta zero input.")
#     end


#     if nchains==0
#         # Initialize output storage
#         thetas = _make_storage(theta0, niter, nburnin, nthin)
#         blobs = _make_storage(blob0, niter, nburnin, nthin)
#         return p0, theta0, blob0, thetas, blobs, nchains, pdftype
#     else
#         # Initialize output storage
#         thetas = _make_storage(theta0s[1], niter, nburnin, nthin, nchains)
#         blobs = _make_storage(blob0s[1], niter, nburnin, nthin, nchains)

#         if make_SharedArray
#             p0s, theta0s, thetas = map(x->convert(SharedArray,x), (p0s, theta0s, thetas))
#             if blobs!=nothing
#                 blob0s, blobs = map(x->convert(SharedArray,x), (blob0s, blobs))
#             end
#         end
#         return p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype
#     end
# end



# ## TODO:
# # - Metropolis-Hastings
# # - block-wise

# ####
# # The MC hammer: emcee implementation
# ###
# # https://github.com/dfm/emcee

# """
# Parallel emcee Metropolis sampler: https://github.com/dfm/emcee

# Input:

# - pdf -- The probability density function to sample, defaults to
#          log-pdf.  The likelihood*prior in a Bayesian setting.

#          `pdf` returns the pdf and an arbitrary other thing which gets
#          stored (return `nothing` to store nothing).

#          `p, blob = pdf(theta)` where `theta` are the parameters.
#          Note that blob can use pre-allocated memory as it will be copied
#          (but not deepcopied) before storage.

# - theta0 -- initial parameters.
#   - If a tuple: choose random thetas around theta0[1] in a ball of radius theta0[2]
#   - If a vector of vectors: use given starting points and that number of chains.

# Optional key-word input:

# - nchain -- number of chain to use (10^3).  If theta0 is vector of vectors, then set accordingly.
# - niter -- number of steps to take (10^4)
#            Thus the total number of likelihood! evaluations is nchain*niter
# - nburnin -- number of initial steps discarded, aka burn-in (niter/10)
# - nthin -- only store every n-th sample (default=1)
# - logpdf -- either true  (default) (for log-likelihoods) or false

# Output:

# - samples:
#   - if typeof(theta0)==Vector then Array(eltype(theta0), length(theta0), niter-nburnin)
#   - if typeof(theta0)!=Vector then Array(typeof(theta0), niter-nburnin)
# - blobs: anything else that the pdf-function returns as second argument
# - accept_ratio: ratio accepted to total steps

# Note: use `squash_chains` to concatenate all chains into one chain.

# Reference: emcee: The MCMC hammer, Foreman-Mackey et al. 2013
# """
# function emcee(pdf, theta0;
#                niter=10^4,
#                nburnin=niter÷10,
#                nchains=10^2,
#                nthin=1,
#                logpdf=true,
#                a_scale=2.0, # step scale parameter.  Probably needn't be adjusted
#                )
#     p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype =
#         _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, make_SharedArray=false)
#     # do the MCMC
#     _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter, nburnin, nchains, nthin, pdftype, a_scale)
# end
# function _emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter, nburnin, nchains, nthin, pdftype, a_scale)
#     # initialization and work arrays:
#     naccept = SharedArray{Int}(nchains, init = S->S[:]=0)
#     ni = SharedArray{Int}(nchains, init = S->S[:]=1)
#     N = length(theta0s[1])

#     #@inbounds
#     for n = (1-nburnin):(niter-nburnin)
#         for nc = 1:nchains
#             # draw a random other chain
#             no = rand(1:nchains-1)
#             no = no>=nc ? no+1 : no # shift by one
#             # sample g (eq. 10)
#             z = sample_g(a_scale)

#             # propose new step:
#             theta1 = theta0s[no] + z*(theta0s[nc]-theta0s[no]) # eq. 7
#             # and its pdf:
#             p1, blob1 = pdf(theta1)

#             # if z^(N-1)*p1/p0>rand() then accept:
#             if z^(N-1)*delog(ratio(p1,p0s[nc], pdftype), pdftype)>rand() # ugly because of log & non-log pdfs
#                 theta0s[nc] = theta1
#                 p0s[nc] = p1

#                 blob0s[nc] = blob1
#                 if n>0
#                     naccept[nc] += 1
#                 end
#             end
#             if  n>0 && rem(n,nthin)==0
#                 _setindex!(thetas, theta0s[nc], ni[nc], nc)
#                 _setindex!(blobs, blob0s[nc], ni[nc], nc)
#                 ni[nc] +=1
#             end
#         end # for nc =1:nchains
#     end # for n=(1-nburnin):(niter-nburnin)

#     accept_ratio = [na/(niter-nburnin) for na in naccept]

#     return thetas, blobs, accept_ratio
# end


# "parallel"
# function emceep(pdf, theta0;
#                nchains=10^2,
#                niter=10^4,
#                nburnin=niter÷10,
#                logpdf=true,
#                nthin=1,
#                a_scale=2.0 # step scale parameter.  Probably needn't be adjusted
#                 )
#     p0s, theta0s, blob0s, thetas, blobs, nchains, pdftype =
#         _initialize(pdf, theta0, niter, nburnin, logpdf, nchains, nthin, make_SharedArray=true)
#     nchains<2 && error("Need nchains>1")
#     # do the MCMC
#     _parallel_emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter, nburnin, nchains, nthin, pdftype, a_scale)
# end

# function _parallel_emcee!(p0s, theta0s, blob0s, thetas, blobs, pdf, niter, nburnin, nchains, nthin, pdftype, a_scale)
#     # @show map(typeof, (p0s, theta0s, blob0s, thetas, blobs))
#     # @show pdf, niter, nburnin, nchains, nthin, pdftype, a_scale

#     # initialization and work arrays:
#     naccept = convert(SharedArray, zeros(Int, nchains))
#     ni = convert(SharedArray, ones(Int, nchains))
#     N = length(theta0s[1])

# #    @show nchains, 1:nchains÷2
#     # the two sets
#     nchains12 = UnitRange{Int}[1:nchains÷2, nchains÷2+1:nchains]

# #     @inbounds
#     for n = (1-nburnin):(niter-nburnin)
#         for i=1:2
#             ncs = nchains12[i] # chains to update
#             ncso = nchains12[mod1(i+1,2)] # other chains, to stretch-move with
#             @sync @parallel for nc in ncs
#                 # draw a random other chain
#                 nco = rand(ncso)
#                 # sample g (eq. 10)
#                 z = sample_g(a_scale)

#                 # propose new step with stretch-move:
#                 theta1 = theta0s[nco] + z*(theta0s[nc]-theta0s[nco]) # eq. 7
#                 # and its density:
#                 p1, blob1 = pdf(theta1)

#                 # if z^(N-1)*p1/p0>rand() then accept:
#                 # (ugly because of log & non-log pdfs)
#                 if z^(N-1)*delog(ratio(p1,p0s[nc], pdftype), pdftype)>rand()
#                     theta0s[nc] = theta1
#                     p0s[nc] = p1
#                     blob0s[nc] = blob1
#                     if n>0
#                         naccept[nc] += 1
#                     end
#                 end
#                 if  n>0 && rem(n,nthin)==0
#                     _setindex!(thetas, theta0s[nc], ni[nc], nc)
#                     _setindex!(blobs, blob0s[nc], ni[nc], nc)
#                     ni[nc] +=1
#                 end
#             end # for nc in ncs
#         end # for i=1:2
#     end # for n=(1-nburnin):(niter-nburnin)

#     accept_ratio = [na/(niter-nburnin) for na in naccept]

#     return thetas, blobs, accept_ratio
# end


# "Puts all samples into one vector"
# function squash_chains(thetas, blobs, accept_ratio)
#     if ndims(thetas)==3
#         return thetas[:,:], blobs==nothing ? nothing : blobs[:,:], mean(accept_ratio)
#     else
#         return thetas[:], blobs==nothing ? nothing : blobs[:], mean(accept_ratio)
#     end
# end

# "g-pdf, see eq. 10 of Foreman-Mackey et al. 2013."
# g_pdf(z, a=2.0) = 1/a<=z<=a ? 1/sqrt(z) * 1/(2*(sqrt(a)-sqrt(1/a))) : zero(z)

# """
# Inverse cdf of g-pdf, see eq. 10 of Foreman-Mackey et al. 2013.
# """
# cdf_g_inv(u, a=2.0) = (u*(sqrt(a)-sqrt(1/a)) + sqrt(1/a) )^2

# "Sample from g using inverse transform sampling"
# sample_g(a=2.0) = cdf_g_inv(rand(), a)
