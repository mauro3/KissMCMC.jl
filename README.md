# KissMCMC

[![Build Status](https://travis-ci.org/mauro3/KissMCMC.jl.svg?branch=master)](https://travis-ci.org/mauro3/KissMCMC.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/mauro3/KissMCMC.jl?branch=master&svg=true)](https://ci.appveyor.com/project/mauro3/kissmcmc-jl/branch/master)

**NOTE: this does not work yet...**

Got a probability density function you want to draw samples from?
Don't want to learn all the fancy stuff of the fancy sampler packages?
The KissMCMC (Keep it simple, stupid, MCMC) package intends to provide
a few simple MCMC samplers (and also a few other samplers).

```julia
using KissMCMC
# the distribution to sample from,
logpdf{T}(x::T) = x<0 ? -convert(T,Inf) : -x
# initial point of chain
theta0 = 0.5

# Metropolis MCMC sampler:
sample_prop_normal(theta) = 1.5*randn() + theta # samples the proposal (or jump) distribution
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
```
outputs:

![](https://cloud.githubusercontent.com/assets/4098145/16770344/dcb4a47a-484c-11e6-8f6e-0c2d223e9443.png)

MCMC samplers:

- Metropolis (serial and parallel) `metropolis` & `metropolisp`
- emcee (serial and parallel) `emcee` & `emceep`

Other serial samplers:

- inverse transform sampler `inverse_transform_sample`
- rejection sampler `rejection_sample_unif` & `rejection_sample`

# TODO

- improve parallel samplers
- refactor serial and parallel samplers into one?
- add convergence tests: autocorrelation, autocovariance, Gelman-Rubin
- tests: better ways to test convergence for Rosenbrock pdf

# References

Other, probably better Julia MCMC packages:

- https://github.com/JuliaStats/Lora.jl
- https://github.com/brian-j-smith/Mamba.jl
- https://github.com/QuantifyingUncertainty

Other packages:

- original emcee python package: http://dan.iel.fm/emcee/current/

Misc references:

- https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
- https://theclevermachine.wordpress.com/2012/11/04/mcmc-multivariate-distributions-block-wise-component-wise-updates/
