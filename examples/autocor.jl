# This follows https://emcee.readthedocs.io/en/latest/tutorials/autocorr/

using KissMCMC, Plots, StatPlots, MCMCDiagnostic
pyplot()

# double bumps pdf in 3D
log_prob(p) = log(exp(-0.5*sum(p.^2)) + exp(-0.5*sum((p-4.0).^2)))


# as on webpage
nwalkers = 32
niter = nwalkers * [5*10^5, 10^6][1]
burnin = 0 # no burnin! on web-page
thetas, accept_ratio, _, logposteriors = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=burnin)

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

#################

# How does this compare to MCMCDiagnostic.effective_sample_size?
#
# help?> MCMCDiagnostics.effective_sample_size
#   effective_sample_size(x, v = var(x))

#   Effective sample size of vector x.

#   Estimated from autocorrelations. See Gelman et al (2013), section 11.4.

#   When the variance v is supplied, it saves some calculation time.

chain = thetas[1, :, 1];
# τ for both:
mcmcd = []
kissd = []
N = round(Int, logspace(2,log10(length(chain)), 10))
for n in N
    push!(mcmcd, n/MCMCDiagnostics.effective_sample_size(chain[1:n]))
    push!(kissd, n/eff_samples(reshape(chain[1:n], (1, length(chain[1:n]), 1)))[1])

    push!(mcmcd, MCMCDiagnostics.autocorrelations(chain[1:n]))
    push!(kissd, n/eff_samples(reshape(chain[1:n], (1, length(chain[1:n]), 1)))[1])

end
plot(N, mcmcd, label="mcmcd")
plot!(N, kissd, label="kiss", xlabel="chain length", ylabel="estimated τ", xscale=:log10, yscale=:log10)


#################
# Check influence of chain size on R_hat
# do as in Gelman et al 2014, p 285

# more independent chain
thetas2, accept_ratio2, _, logposteriors2 = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=burnin)
thetas3, accept_ratio3, _, logposteriors3 = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=burnin)
thetas4, accept_ratio4, _, logposteriors4 = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=burnin)
thetas5, accept_ratio5, _, logposteriors5 = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=burnin)
ths = [thetas, thetas2, thetas3, thetas4, thetas5]
N = round(Int, logspace(2,log10(length(chain)), 10))
walker = 1


index = 1
NN = []
Rhat = []
for n in N
    burnin = n÷2
    chains = [t[index, burnin:n, walker] for t in ths]
    push!(Rhat, MCMCDiagnostics.potential_scale_reduction(chains...))
    push!(NN, n÷2)
end
plot(NN, Rhat, xscale=:log10)
