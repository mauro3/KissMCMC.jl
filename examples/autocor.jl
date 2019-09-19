# This follows https://emcee.readthedocs.io/en/latest/tutorials/autocorr/

using KissMCMC, Plots, StatPlots
pyplot()

# double bumps pdf in 3D
log_prob(p) = log(exp(-0.5*sum(p.^2)) + exp(-0.5*sum((p-4.0).^2)))


# as on webpage
nwalkers = 32
niter = nwalkers * [5*10^5, 10^6][1]
# no burnin!
thetas, accept_ratio, _, logposteriors = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=0)
chain = thetas[1:1, :, :];
#chain = thetas[1, :, :];
#histogram(chain[:], xlabel="θ", ylabel="p(θ)")

N = round(Int, logspace(2,log10(size(thetas,2)), 10))

c = 10 # window width default is 5
nthin = 10 # what happens with thinned results? -> all ok
taus = []
converged = []
NN = []
for n in N
    if length(1:nthin:n)>3
        tau, conv = KissMCMC.int_acorr(chain[:, 1:nthin:n, :], warn=false)
        push!(taus, tau[1])
        push!(converged, conv)
        push!(NN,length(1:nthin:n))
    end
end

plot(NN, taus,
     xscale=:log10,
     yscale=:log10,
     xlabel="Number of samples N",
     ylabel="τ estimate",
     label="DFM 2017",
     ticks=:native,
     reuse=false,
     title="Using $(size(chain,3)) chains")
plot!(NN, NN/50, label="N/50", color=:black, ls=:dash)

# What happens when the chains are concatenated?
# This seems to work also.  I suspect this only works at longer chain lengths.
#
# Testing this on "real" data gave less promising results...
thetass = squash_chains(thetas)[1]
chain = reshape(thetass[1,:], (1, size(thetass,2), 1))

taus = []
converged = []
NN = []
for n in round(Int, logspace(2,log10(length(chain)), 10))
    if length(1:nthin:n)>3
        tau, conv = KissMCMC.int_acorr(chain[:, 1:nthin:n, :], warn=false)
        push!(taus, tau[1])
        push!(converged, conv)
        push!(NN,length(1:nthin:n))
    end
end
plot(NN, taus,
     xscale=:log10,
     yscale=:log10,
     xlabel="Number of samples N",
     ylabel="τ estimate",
     label="DFM 2017",
     ticks=:native,
     reuse=false,
     title="Using $(size(chain,3)) chains")
plot!(NN, NN/50, label="N/50", color=:black, ls=:dash)
