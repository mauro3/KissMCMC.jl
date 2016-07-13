# A simple example doing Bayesian inference using MCMC and my
# samplers.  This example assumes that the times of measurements are
# errror-free.

# this is where the forward model, probabilistic model etc are defined.
include("bayesian-example-base.jl")
plotyes = false

# make the synthetic data
sigma = 0.3
const ts, data, A, ω, ϕ, ts_true, data_true = make_data(para_true = [2.5, 1.1, 3], # [A, ω, ϕ]
                                                        sigma_x=sigma,
                                                        sigma_v=sigma,
                                                        sigma_t=0)
@assert all(ts.==ts_true)

## Plot the data
# using Plots,StatPlots
# plotdata(ts,data,A,ω,ϕ)

# choose to use the analytic or ODE-forward model
fwd! = [fwd_ana!, fwd_ode!][1]
const fwdout = init_fwd(ts) # Note, this will be modified in place

##############################
# Parameter estimation using the MCMC functions
##############################
#
# fitting parameters [A,ω,ϕ,sigma] = theta

# Likelilhood

function loglikelihood(theta)
    # Closure over fwdout, fwd!, data, ts
    # make sure those are `const`!
    A,ω,ϕ,sigma = theta
    fwd!(fwdout, ts, A,ω,ϕ)
    loglikelihood(fwdout, data, sigma)
end

# Normal & uniform priors
logprior_A(A) = 0<=A ? 0.0 : -Inf # amplitude is positive
ω_max = 15.0
logprior_ω(ω) = 0<=ω<ω_max ? 0.0 : -Inf # ω is bounded
logprior_ϕ(ϕ) = 0<=ϕ<2*pi ? 0.0 : -Inf # ϕ is bounded

sigma_est = 0.2 # our prior estimate of sigma
sigma_est_sigma = 0.2 # our estimate of the std of sigma
logprior_sigma(sigma) = sigma[1]<=0 ? -Inf : -(sigma[1]-sigma_est)^2/(2*sigma_est_sigma)

logprior = (theta) -> (logprior_A(theta[1]) +
                       logprior_ω(theta[2]) +
                       logprior_ϕ(theta[3]) +
                       logprior_sigma(theta[4]) )

logposterior = @anon theta -> loglikelihood(theta) + logprior(theta)

######
# MCMC
######
nthin = 10
niter = 5*10^5
nburnin = niter÷10
nchains = 50
niter_e = niter÷nchains
nburnin_e = niter_e÷2

#################
# Metropolis MCMC
#################

# Using a Gaussian proposal distribution.  Note, this needs to be
# symmetric for Metropolis!

sig = 0.01  # this needs tuning, accept_ratio of 1/4 is good, they say.
const sigma_ppdf = [sig, sig, sig, sig]
function sample_ppdf(thetas)
    A,ω,ϕ,sigma = thetas
    out = similar(thetas)
    out[1] = randn()*sigma_ppdf[1]+A
    out[2] = randn()*sigma_ppdf[2]+ω
    out[3] = randn()*sigma_ppdf[3]+ϕ
    out[4] = randn()*sigma_ppdf[4]+sigma
    return out
end

thetas0 = [A, ω, ϕ, sigma];  # good IC
thetas0 = [2.1, 1.1, 1.1, 0.2]; # decent IC
# note that the emcee sampler does poorly with a poor IC!
#thetas0 = [1.74,0.001, 0.25, 1.77] # bad IC

@time 1
metropolis(logposterior, sample_ppdf, thetas0, niter=2)
print("Metropolis: ")
@time thetas_m, accept_ratio_m = metropolis(logposterior, sample_ppdf, thetas0,
                                            niter=niter, nthin=nthin, nburnin=nburnin)
print_results("Metropolis", [A,ω,ϕ,sigma], thetas_m, accept_ratio_m)

emcee(logposterior, (thetas0, 0.1), niter=10, nchains=2)
print("emcee:")
@time thetas_ec, accept_ratio_ec = emcee(logposterior, (thetas0, 0.1),
                                         niter=niter_e, nthin=nthin, nchains=nchains, nburnin=nburnin_e)
# When running this problem with IC far from the maximum, then emcee produces
thetas_e, accept_ratio_e = squash_chains(thetas_ec, accept_ratio_ec, drop_low_accept_ratio=true)
print_results("emcee", [A,ω,ϕ,sigma], thetas_e, accept_ratio_e)

# using Plots,StatPlots # needs current master of StatPlots & on 0.5 of PyCall too (14 July 2016)
#cornerplot(thetas_m[1:3,:]', label=["A","ω", "ϕ"])

nothing
