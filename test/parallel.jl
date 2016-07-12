            ## MCMC parallel
        n_workers = max(length(procs())-1,1)
        println("Parallel MCMC; number of workers: $n_workers")

        # make one chain per worker
        theta0 = n_workers==1 ? [0.1] : linspace(0.1,0.9, n_workers)

        print("Metropolisp    : ")
        metropolisp(pdfblob, sample_prop_normal, theta0, niter=10, logpdf=true)
        @time thetasp, blobsp, accept_ratiop = metropolisp(pdfblob, sample_prop_normal, theta0, niter=n÷n_workers)
        thetasp, blobsp, accept_ratiop = emcee_squash(thetasp, blobsp, accept_ratiop );
        test_mean_std(sa, thetasp)

        print("emceep          : ")
        emceep(pdfblob, (0.5, 0.1), niter=10, nchains=10);
        @time thetas_ep, blobs_ep, accept_ratio_ep = emceep(pdfblob, (0.5, 0.1), niter=n÷nchains, nchains=nchains);
        thetas_ep, blobs_ep, accept_ratio_ep = emcee_squash(thetas_ep, blobs_ep, accept_ratio_ep );
        test_mean_std(sa, thetas_ep)



# Test the samplers against known distributions.
using Compat
using Base.Test
# using Plots
@everywhere include("samplers-parallel.jl")

if !isdefined(:dimensions)
    dimensions = 1
    @show dimensions
end
if !isdefined(:plotyes)
    plotyes = true
end

function plothist(sa, bins, lab)
    histogram(sa, bins, label=lab, histtype="step", normed=true)
    # hsa = hist(sa,bins)
    # plot(hsa[1][1:end-1]+diff(hsAa[1])/2, hsa[2]/maximum(hsa[2]), label=lab)
end
function plothist2(sa, bins, lab)
    scatter(sa[1,:], sa[2,:], label=lab)
    # hsa = hist(sa,bins)
    # plot(hsa[1][1:end-1]+diff(hsAa[1])/2, hsa[2]/maximum(hsa[2]), label=lab)
end

# 1D test
function test_mean_std(s1,s2,diff=0.05)
    if ! (( abs(mean(s1)-mean(s2)) / max(abs(mean(s1)),0.01) < diff) && ( abs(std(s1)-std(s2)) / max(abs(std(s1)),0.01) < diff))
        warn("off!")
    end
end
# 2D test
# function test_mean_std(truedist::Distribution, samples, diff=0.2)
#     Dist = typeof(truedist).name.primary
#     fitted = fit(Dist, samples)
#     @test all(1-diff .< mean(truedist)./mean(fitted) .< 1+diff)
#     @test all(1-diff .< cov(truedist)./cov(fitted) .< 1+diff)
# end



if dimensions==1
    # number of samples to draw
    #    n = 10^7
    n = 10^5

    # our mystery (unscaled) distribution:
    @everywhere rand_025() = rand()-0.25
    @everywhere rand_025(n) = [rand_025() for i=1:n]

    @everywhere immutable ExpPDF end
    @everywhere @compat (::ExpPDF)(x) = x<0 ? -Inf : -x
    ## artificial slow-down
    # n = 5000
    # @everywhere @compat (::ExpPDF)(x) = (sleep(0.001); x<0 ? -Inf : -x)

    @everywhere tests = [(randexp, ExpPDF(), 0:0.1:10)]
    @everywhere if !isdefined(:dist_log)
        const builtin, dist_log, bins = tests[1] # need const to make nested anon fast
    end

    # samples drawn with built-in sampler
    print("Built-in sampler: ")
    @time sa = builtin(n)

    # MCMC
    @everywhere const sigma = 2.0
    @everywhere immutable SampleNormal end
    @everywhere @compat (::SampleNormal)(x0) = randn()*sigma-x0

    # metro
    @everywhere immutable PDF end
    @everywhere @compat (::PDF)(x) = (dist_log(x), nothing)
    @show    nwrk = max(length(procs())-1,1)
    @show nwrk==1
    theta0 = nwrk==1 ? [0.1] : linspace(0.1,0.9, nwrk)
    # print("Metropolis: ")
    # @time thetas, blobs, accept_ratio = metropolis(pdf_blob, sample_prop_normal, theta0, niter=n÷length(theta0), logpdf=false)
    # thetas, blobs, accept_ratio = emcee_squash(thetas, blobs, accept_ratio );
    # @test test_mean_std(sa, thetas)

    print("Metropolis (log): ")
    # warmup:
    thetas, blobs, accept_ratio = metropolisp(PDF(), SampleNormal(), theta0, niter=10)
    @time thetas, blobs, accept_ratio = metropolisp(PDF(), SampleNormal(), theta0, niter=n÷length(theta0))
    thetas, blobs, accept_ratio = emcee_squash(thetas, blobs, accept_ratio );
    @show length(thetas)/10^6
    test_mean_std(sa, thetas)

    print("emcee         :  ")
    thetas_e, blobs_e, accept_ratio_e = emcee(PDF(), (0.5, 0.1), niter=10, nchains=2);
    @time thetas_e, blobs_e, accept_ratio_e = emcee(PDF(), (0.5, 0.1), niter=n÷100, nchains=100);
    thetas_e, blobs_e, accept_ratio_e = emcee_squash(thetas_e, blobs_e, accept_ratio_e );
    @show length(thetas_e)/10^6
    test_mean_std(sa, thetas_e)

    thetas_ep, blobs_ep, accept_ratio_ep = emceep(PDF(), (0.5, 0.1), niter=10, nchains=2);
    print("emcee parallel (1000w):  ")
    @time thetas_ep, blobs_ep, accept_ratio_ep = emceep(PDF(), (0.5, 0.1), niter=n÷1000, nchains=1000);
    print("emcee parallel (100w):  ")
    @time thetas_ep, blobs_ep, accept_ratio_ep = emceep(PDF(), (0.5, 0.1), niter=n÷100, nchains=100);
    thetas_ep, blobs_ep, accept_ratio_ep = emcee_squash(thetas_ep, blobs_ep, accept_ratio_ep );
    @show length(thetas_ep)/10^6
    test_mean_std(sa, thetas_ep)


end
nothing
