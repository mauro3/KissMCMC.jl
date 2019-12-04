# KissMCMC

[![Build Status](https://travis-ci.org/mauro3/KissMCMC.jl.svg?branch=master)](https://travis-ci.com/mauro3/KissMCMC.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mauro3/KissMCMC.jl?svg=true)](https://ci.appveyor.com/project/mauro3/KissMCMC-jl)
[![Codecov](https://codecov.io/gh/mauro3/KissMCMC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/mauro3/KissMCMC.jl)
[![Build Status](https://api.cirrus-ci.com/github/mauro3/KissMCMC.jl.svg)](https://cirrus-ci.com/github/mauro3/KissMCMC.jl)

Got a probability density function you want to draw samples from?
Don't want to learn all the fancy stuff of the fancy sampler packages?
The KissMCMC (Keep it simple, stupid, MCMC) package intends to provide
a few simple MCMC samplers.

```julia
using KissMCMC
# the distribution to sample from,
logpdf(x::T) where {T} = x<0 ? -convert(T,Inf) : -x
# initial point of walker
theta0 = 0.5

# Metropolis MCMC sampler:
sample_prop_normal(theta) = 1.5*randn() + theta # samples the proposal (or jump) distribution
thetas, accept_ratio = metropolis(logpdf, sample_prop_normal, theta0, niter=10^5)
println("Accept ratio Metropolis: $accept_ratio")

# emcee MCMC sampler:
thetase, accept_ratioe = emcee(logpdf, make_theta0s(theta0, 0.1, logpdf, 100), niter=10^5)
# check convergence using integrated autocorrelation
thetase, accept_ratioe = squash_walkers(thetase, accept_ratioe) # puts all walkers into one
println("Accept ratio emcee: $accept_ratio")

using Plots
histogram(thetas, normalize=true, fillalpha=0.4)
histogram!(thetase, normalize=true, fillalpha=0.1)
plot!(0:0.01:5, map(x->exp(logpdf(x)[1]), 0:0.01:5), lw=3)
```
outputs:

![](https://cloud.githubusercontent.com/assets/4098145/16770344/dcb4a47a-484c-11e6-8f6e-0c2d223e9443.png)

MCMC samplers:
- Metropolis (serial) `metropolis`
- Affine invariant MCMC, aka emcee `emcee` (threaded)

# References

Other, probably better Julia MCMC packages:

- https://github.com/tpapp/DynamicHMC.jl
- https://github.com/madsjulia/AffineInvariantMCMC.jl
- https://github.com/brian-j-smith/Mamba.jl
- and many others

The (original) emcee python package: http://dan.iel.fm/emcee/current/
