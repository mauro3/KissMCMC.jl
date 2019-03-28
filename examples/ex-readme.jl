# example from the readme

using KissMCMC
# the distribution to sample from,
logpdf(x::T) where {T} = x<0 ? -convert(T,Inf) : -x
# initial point of chain
theta0 = 0.5

# Metropolis MCMC sampler:
sample_prop_normal(theta) = 1.5*randn() + theta # sample of the proposal (or jump) distribution:
thetas, accept_ratio = metropolis(logpdf, sample_prop_normal, theta0, niter=10^5)
println("Accept ratio Metropolis: $accept_ratio")

# emcee MCMC sampler:
thetase, accept_ratioe = emcee(logpdf, (theta0,0.1), niter=10^4, nchains=10)
thetase, accept_ratioe = squash_chains(thetase,accept_ratioe) # puts all chains into one
println("Accept ratio emcee: $accept_ratioe")

using Plots
histogram(thetas, normalize=true, fillalpha=0.4)
histogram!(thetase, normalize=true, fillalpha=0.1)
plot!(0:0.01:5, map(x->exp(logpdf(x)[1]), 0:0.01:5), lw=3)
