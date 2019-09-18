# this follows https://dfm.io/posts/autocorr/, "A more realistic example"
#
# Docs https://emcee.readthedocs.io/en/latest/user/sampler/

using KissMCMC, Plots, StatPlots
pyplot()

# double bumps pdf in 3D
log_prob(p) = log(exp(-0.5*sum(p.^2)) + exp(-0.5*sum((p-4.0).^2)))


# as on webpage
nwalkers = 32
niter = nwalkers * [5*10^5, 5*10^6][1]
# no burnin!
thetas, accept_ratio, _, logposteriors = emcee(log_prob, ([0.0, 0.0, 0.0], 0.1),
                                               nchains=nwalkers, niter=niter,
                                               nburnin=0)
chain = thetas[1:1, :, :];
#chain = thetas[1, :, :];
#histogram(chain[:], xlabel="θ", ylabel="p(θ)")

N = round(Int, logspace(2,log10(size(thetas,2)), 10))

c = 5 # window width default is 5
taus = []
for n in N
    push!(taus, KissMCMC.intacor(chain[:, 1:n, :], c)[1])
#    push!(taus, KissMCMC.intacor(chain[1:n, :]))
end

plot(N, taus,
     xscale=:log10,
     yscale=:log10,
     xlabel="Number of samples N",
     ylabel="τ estimate",
     label="DFM 2017",
     ticks=:native)
plot!(N, N/50, label="N/50", color=:black, ls=:dash)
