
using KissMCMC
import Distributions, StatsBase
using Parameters: @unpack
const D = Distributions
const SB = StatsBase
using Test, Statistics


std2(x) = sqrt.(var(x)) # works in multi dims

"""
Struct to hold a test-case for MCMC sampling
"""
Base.@kwdef struct ATest
    dist # a Distribution dist or a logpdf function
    dim = length(mean(dist))
    mean_ = mean(dist)
    median_ = dim==1 ? median(dist) : nothing
    std_ = std2(dist)
    skewness_ = dim==1 ? D.skewness(dist) : nothing #
    theta0 = 0.0
    ball_radius = 0.1
    nwalkers = 100
    niter = 10^4
    hasblob = false
    sample_ppdf = theta -> randn() + theta
    tole = 0.3
    tolm = 0.3
end

# helpers
function test_mean_std(thetas, atest::ATest, tol)
    @unpack mean_, std_, skewness_, median_ = atest
    @show atest.dist
    mean_!=nothing && @test all(abs.(mean(thetas) .- mean_) .< abs.(std_ * tol))
    std_!=nothing && @test all(abs.(std(thetas) - std_) .< abs.(std_ * tol))
    median_!=nothing && @test abs.(median(thetas) - median_) < abs.(std_ * tol)
    skewness_!=nothing && @test abs.(SB.skewness(thetas) - skewness_) < abs.(std_ * 2*tol)
end
function test_blobs(truths::Dict, blobs)
    nothing
    # TODO
end
function test_logdensities(samples, logdensities, pdf)
    nothing
end

# TODO
#             # Rosenbrock
#             rosenpdf =  x-> -( 100*(x[2]-x[1]^2)^2+(1-x[1])^2 )/20

testcases = [
    ATest(dist = D.Normal(-5.0, 3.0),
          theta0=-4.0,
          sample_ppdf = theta -> 9*randn() + theta,
          ),
    ATest(dist = D.LogNormal(0.0,1.0),
          theta0 = 0.4,
          sample_ppdf = theta -> 7.5*randn() + theta,
          niter = 10^7,
          tolm = 0.4),
    ATest(dist = D.MvNormal([0.5,-0.25], [0.47 1.8
                                          1.8 7.]), # 2D normal dist
          theta0 = [0.4, 0.3],
          sample_ppdf = theta -> 0.5*randn(2) .+ theta,
          niter = 10^5,
          ),
    ATest(dist =  x-> -(100*(x[2] - x[1]^2)^2 + (1 - x[1])^2 ) / 20, # Rosenbrock
          dim = 2,
          mean_ = [0.98, 10.3], # from running emcee with niter=10^9
          median_ = nothing,
          std_ = [3.1, 13.8],
          skewness_ = nothing,
          theta0 = [0.0, 0.0],
          niter = 10^7,
          sample_ppdf = theta -> 0.5*randn(2) .+ theta,
          tolm = 0.6,
          tole = 0.6),]

make_lpdf(dist::D.Distribution) = x -> D.logpdf(dist, x)
make_lpdf(dist) = dist

## run tests
include("metro.jl")
include("emcee.jl")
