# A simple example doing Bayesian inference for a harmonic oscillator
# using MCMC and the KissMCMC samplers.
#
# The forward model and other bits are defined in
# bayesian-example-base.jl.
#
# This example assumes that the times of measurements are errror-free.

# this is where the forward model, probabilistic model etc are defined.
include("bayesian-example-base.jl")
plotyes = false
if plotyes
    eval(:(using Plots,StatPlots)) # needs current master of StatPlots & on 0.5 of PyCall too (14 July 2016)
end

# make the synthetic xv_measured
sigma = 0.3
const ts_measured, xv_measured, A, ω, ϕ, ts_true, xv_true = make_synthetic_measurements(theta_true = [2.5, 1.1, 3], # [A, ω, ϕ]
                                                                               sigma_x=sigma,
                                                                               sigma_v=sigma,
                                                                               sigma_t=0)
@assert all(ts_measured.==ts_true)

## Plot the measurements

# using Plots,StatPlots
plotyes && plotmeasurements(ts_measured,xv_measured,A,ω,ϕ)

# choose to use the analytic or ODE-forward model
fwd! = [fwd_ana!, fwd_ode!][1]
const fwdout = init_fwd(ts_measured) # Note, this will be modified in place

##############################
# Parameter estimation using the MCMC functions
##############################
#
# fitting parameters [A,ω,ϕ,sigma] = theta

varnames = ["A","ω", "ϕ", "σ"]

# Likelilhood

function loglikelihood(theta)
    # Closure over fwdout, fwd!, xv_measured, ts_measured
    # make sure those are `const`!
    A,ω,ϕ,sigma = theta
    fwd!(fwdout, ts_measured, A,ω,ϕ)
    loglikelihood(fwdout, xv_measured, sigma)
end

# Normal & uniform priors
logprior_A(A) = 0<=A ? 0.0 : -Inf # amplitude is positive
ω_max = 15.0
logprior_ω(ω) = 0<=ω<ω_max ? 0.0 : -Inf # ω is bounded
logprior_ϕ(ϕ) = 0<=ϕ<2*pi ? 0.0 : -Inf # ϕ is bounded

sigma_est = 0.2 # our prior estimate of sigma
sigma_est_sigma = 0.2 # our estimate of the std of sigma
logprior_sigma(sigma) = sigma<=0 ? -Inf : -(sigma-sigma_est)^2/(2*sigma_est_sigma)

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
sample_ppdf(theta) = [randn()*sigma_ppdf[i]+theta[i] for i=1:length(theta)]

theta_true = [A, ω, ϕ, sigma];  # good IC
theta0 = [2.1, 1.1, 1.1, 0.2]; # decent IC
# note that the emcee sampler does poorly with a poor IC!
#theta0 = [1.74,0.001, 0.25, 1.77] # bad IC

@time 1
metropolis(logposterior, sample_ppdf, theta0, niter=2)
print("Metropolis: ")
@time thetas_m, accept_ratio_m = metropolis(logposterior, sample_ppdf, theta0,
                                            niter=niter, nthin=nthin, nburnin=nburnin)
print_results(thetas_m, accept_ratio_m, names=varnames, title="Metropolis", theta_true=theta_true)

emcee(logposterior, (theta0, 0.1), niter=10, nchains=2)
print("emcee:")
@time thetas_ec, accept_ratio_ec = emcee(logposterior, (theta0, 0.1),
                                         niter=niter_e, nthin=nthin, nchains=nchains, nburnin=nburnin_e)
# When running this problem with IC far from the maximum, then emcee produces
thetas_e, accept_ratio_e = squash_chains(thetas_ec, accept_ratio_ec, drop_low_accept_ratio=true)
print_results(thetas_e, accept_ratio_e, names=varnames, title="emcee", theta_true=theta_true)

plotyes && cornerplot(thetas_m[1:3,:]', label=["A","ω", "ϕ"])

nothing
