# Test the samplers against known distributions.

using Base.Test
using Plots
@everywhere import Distributions
using FastAnonymous
const Dist = Distributions
@everywhere include("samplers-v2.jl")

if !isdefined(:dimensions)
    dimensions = 1
    @show dimensions
end
if !isdefined(:plotyes)
    plotyes = true
end

function plothist(sa, bins, lab)
    plt[:hist](sa, bins, label=lab, histtype="step", normed=true)
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
    err_mean = abs(mean(s1)-mean(s2)) / max(abs(mean(s1)),0.01)
    err_std = abs(std(s1)-std(s2)) / max(abs(std(s1)),0.01)
    if err_mean > diff
        warn("Mean is off my more than $(diff*100)%! err_mean=$err_mean")
    end
    if err_std > diff
        warn("Std is off my more than $(diff*100)%! err_std=$err_std")
    end
end
function test_mean_std(s1::Matrix,s2::Matrix, diff=0.05)
    for i=1:size(s1,1)
        err_mean = abs(mean(s1[i,:])-mean(s2[i,:])) / max(abs(mean(s1[i,:])),0.01)
        err_std = abs(std(s1[i,:])-std(s2[i,:])) / max(abs(std(s1[i,:])),0.01)
        if err_mean > diff
            warn("Mean is off my more than $(diff*100)%! err_mean=$err_mean in row $i")
        end
        if err_std > diff
            warn("Std is off my more than $(diff*100)%! err_std=$err_std in row $i")
        end
    end
end

# 2D test
# function test_mean_std(truedist::Dist.Distribution, samples, diff=0.2)
#     D = typeof(truedist).name.primary
#     fitted = Distributions.fit(D, samples)
#     @test all(1-diff .< mean(truedist)./mean(fitted) .< 1+diff)
#     @test all(1-diff .< cov(truedist)./cov(fitted) .< 1+diff)
# end



if dimensions==1
    # number of samples to draw
    n = 5*10^6
    n = Int(10^5) #6
    nchains = max(min(1000, n÷10^4),100)


    # our mystery (unscaled) distribution:
    @everywhere begin
        rand_025() = rand()-0.25
        rand_025(n) = [rand_025() for i=1:n]

        # FastAnonymous does not work in parallel, thus hand-code functors
        immutable UnifPDF end
        call(::UnifPDF, x) = -0.25<=x<=0.75 ? 0.0:-Inf
        unif = UnifPDF()

        immutable NormPDF end
        call(::NormPDF, x) = -x^2/2
        npdf = NormPDF()

        immutable ExpPDF end
        call(::ExpPDF, x) = x<0 ? -Inf : -x
        exppdf = ExpPDF()

        "Does nothing but slow down.  Uses ~sec seconds to complete on my machine."
        function slowdown(sec)
            out = 0.0
            n = Int(2*sec*10^8)
            # 10^5 ~ 0.001s
            for i=1:n
                out += rand()
            end
            out
        end

        immutable SlowExpPDF end
        call(::SlowExpPDF, x) = (slowdown(0.001); x<0 ? -Inf : -x)
        slowexppdf = SlowExpPDF()

        tests = [(rand_025  , unif, -0.5:0.01:1),
                 (randn, npdf , -5:0.1:5), # normal, bins
                 (randexp, exppdf, 0:0.1:10),
                 (randexp, slowexppdf, 0:0.1:10)]
        if !isdefined(:dist)
            const builtin, pdf_log, bins = tests[3] # need const to make nested closures fast

            immutable PDF end
            call(::PDF, x) = exp(pdf_log(x))
            const pdf = PDF()
        end
    end

    # samples drawn with built-in sampler
    print("Built-in sampler: ")
    builtin(1)
    @time sa = builtin(n)

    # # rejection sampling
    # print("Uniform rejection sampler: ")
    # @time sa_rej = Float64[rejection_sample_unif(pdf, [minimum(bins),maximum(bins)], 1) for i=1:n]
    # test_mean_std(sa, sa_rej)

    # # rejection sampling v2
    # print("General rejection sampler: ")
    # f = @anon x->2*pdf(x) # proposal distribution
    # g = @anon () -> builtin()
    # @time sa_rej2 = Float64[rejection_sample(pdf, f, g) for i=1:n]
    # test_mean_std(sa, sa_rej2)

    # # MCMC
    # sample PDF
    @everywhere begin
        const sigma = 2.0
        immutable SampleNormal end
        call(::SampleNormal, x0) = randn()*sigma+x0
        sample_prop_normal = SampleNormal()

        # PDF
        immutable PDFblob end
        call(::PDFblob, x) = (pdf_log(x), nothing)
        pdfblob = PDFblob()
    end

    println("Serial MCMC")
    # # warmup
    metropolis(pdfblob, sample_prop_normal, 0.5, niter=10, logpdf=true)
    print("Metropolis     : ")
    @time thetas, blobs, accept_ratio = metropolis(pdfblob, sample_prop_normal, 0.5, niter=n)
    test_mean_std(sa, thetas)
    #@show length(thetas)/10^6


    emcee(pdfblob, (0.5, 0.1), niter=10, nchains=10);
    print("emcee           : ")
    @time thetas_e, blobs_e, accept_ratio_e = emcee(pdfblob, (0.5, 0.1), niter=n÷100, nchains=100);
    thetas_e, blobs_e, accept_ratio_e = emcee_squash(thetas_e, blobs_e, accept_ratio_e );
    test_mean_std(sa, thetas_e)


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


elseif dimensions==2
    #### Multivariate distributions
    # number of samples to draw
    n = 5*10^5
    nchains = max(min(1000, n÷10^4),100)

    @everywhere begin
        # narrow Normal
        const mu = [0.5,-0.25]
        const covm = [0.47 1.8
                      1.8 7.]
        const icovm = inv(covm)
        mvn = Distributions.MvNormal(mu, covm)

        immutable NormPDF2 end
        call(::NormPDF2, x) = (-(x-mu)'*icovm*(x-mu)/2)[1]
        npdf2 = NormPDF2()

        # Rosenbrock
        immutable RosenPDF end
        call(::RosenPDF, x) = -( 100*(x[2]-x[1]^2)^2+(1-x[1])^2 )/20
        rosenpdf = RosenPDF()

        tests = Vector{Any}[Any[
                                @anon(() -> rand(mvn)),
                                @anon((n) -> rand(mvn,n)),
                                npdf2,
                                FloatRange{Float64}[-3:0.1:4, -12:0.1:12]
                                ],
                            Any[
                                @anon(() -> NaN), # no analytic available
                                @anon((n) -> NaN*ones(n)),
                                rosenpdf,
                                FloatRange{Float64}[-4:0.1:6, -1:0.1:31]
                                ]
                            ]
        if !isdefined(:pdf_log)
            const builtin, builtin_n, pdf_log, bins = tests[1]

            immutable PDF2 end
            call(::PDF2, x) = exp(pdf_log(x))
            const pdf2 = PDF2()
        end
    end

   # samples drawn with built-in sampler
   print("Built-in sampler: ")
    builtin_n(1)
    @time sa2 = builtin_n(n)

    # rejection sampling
    print("Uniform rejection sampler: ")
    @time sa2_rej = hcat(Vector{Float64}[rejection_sample_unif(pdf2, Vector{Float64}[map(minimum, bins),map(maximum, bins)], 1.0) for i=1:n]...)
    test_mean_std(sa2, sa2_rej)
    f = @anon x->2*pdf2(x)
g = @anon () -> builtin()
    print("General rejection sampler: ")
    @time sa2_rej2 = hcat(Vector{Float64}[rejection_sample(pdf2, f, g) for i=1:n]...)
    test_mean_std(sa2, sa2_rej2)
    #@time sa2_rej2 = Float64[rejection_sample(pdf2, x->2, ()->(rand()-0.5)*10  )  for i=1:n]

    # # MCMC
    # sample PDF
    @everywhere begin
        const sigma = 2.0
        immutable SampleNormal2 end
        call(::SampleNormal2, x0) = randn(2)*sigma-x0
        sample_prop_normal2 = SampleNormal2()

        # PDF
        immutable PDFblob end
        call(::PDFblob, x) = (pdf_log(x), nothing)
        pdfblob = PDFblob()
    end

    println("Serial MCMC")
    # # warmup
    metropolis(pdfblob, sample_prop_normal2,  [0.5, 0.5], niter=10, logpdf=true)
    print("Metropolis     : ")
    @time thetas, blobs, accept_ratio = metropolis(pdfblob, sample_prop_normal2, [0.5, 0.5], niter=n)
    test_mean_std(sa2, thetas)

    emcee(pdfblob, ([0.5,0.5], 0.1), niter=10, nchains=10);
    print("emcee           : ")
    @time thetas_e, blobs_e, accept_ratio_e = emcee(pdfblob, ([0.5,0.5], 0.1), niter=n÷100, nchains=100);
    thetas_e, blobs_e, accept_ratio_e = emcee_squash(thetas_e, blobs_e, accept_ratio_e );
    test_mean_std(sa2, thetas_e)

    ## MCMC parallel
    n_workers = max(length(procs())-1,1)
    println("Parallel MCMC; number of workers: $n_workers")

# make one chain per worker
a = n_workers>1 ? linspace(0.1,0.9, n_workers):nothing
    theta0 = n_workers==1 ?  Vector{Float64}[[0.5, 0.5]] : [[0.5,a[i]] for i=1:n_workers]

    print("Metropolisp    : ")
    metropolisp(pdfblob, sample_prop_normal2, theta0, niter=10, logpdf=true)
    @time thetasp, blobsp, accept_ratiop = metropolisp(pdfblob, sample_prop_normal2, theta0, niter=n÷n_workers)
    thetasp, blobsp, accept_ratiop = emcee_squash(thetasp, blobsp, accept_ratiop );
    test_mean_std(sa2, thetasp)

    print("emceep          : ")
    emceep(pdfblob, ([0.5, 0.5], 0.1), niter=10, nchains=10);
    @time thetas_ep, blobs_ep, accept_ratio_ep = emceep(pdfblob, ([0.5,0.5], 0.1), niter=n÷nchains, nchains=nchains);
    thetas_ep, blobs_ep, accept_ratio_ep = emcee_squash(thetas_ep, blobs_ep, accept_ratio_ep );
    test_mean_std(sa2, thetas_ep)


    # # MCMC
    # sigma = [1.0,1.0]
    # prop_normal = @anon (x,x1) -> pdf(MvNormal(x1, sigma),x)
    # prop_normal1d = @anon (x,x1) -> pdf(Normal(x1, sigma[1]),x)
    # prop_normal_log = @anon (x,x1) -> logpdf(MvNormal(x1, sigma),x)
    # prop_normal1d_log = @anon (x,x1) -> logpdf(Normal(x1, sigma[1]),x)
    # sample_prop_normal = @anon (x0) -> rand(MvNormal(x0, sigma))
    # sample_prop_normal1d = @anon (x0) -> rand(Normal(x0, sigma[1]))

    # # metro
    # niter=n*100
    # @time sa2_m, accept_ratio = mcmc_m(dist, sample_prop_normal, [0.5,0.5], niter=niter)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_m)
    # @time sa2_m_log, accept_ratio = mcmc_m(pdf_log, sample_prop_normal, [0.5,0.5], niter=niter, logpdf=true)
    # @show accept_ratio
    #     test_mean_std(mvn, sa2_m_log)
    # @time sa2_m_cw, accept_ratio = mcmc_m_cw(dist, [sample_prop_normal1d], [0.5,0.5], niter=niter)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_m_cw)
    # @time sa2_m_cw_log, accept_ratio = mcmc_m_cw(pdf_log, [sample_prop_normal1d], [0.5,0.5], niter=niter, logpdf=true)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_m_cw_log)

    # # metro-hast
    # @time sa2_mh, accept_ratio = mcmc_mh(dist, prop_normal, sample_prop_normal, [0.5,0.5], niter=niter)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_mh)
    # @time sa2_mh_log, accept_ratio = mcmc_mh(pdf_log, prop_normal_log, sample_prop_normal, [0.5,0.5], niter=niter, logpdf=true)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_mh_log)

    # @time sa2_mh_cw, accept_ratio = mcmc_mh_cw(dist, [prop_normal1d], [sample_prop_normal1d], [0.5,0.5], niter=niter)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_mh_cw)
    # @time sa2_mh_cw_log, accept_ratio = mcmc_mh_cw(pdf_log, [prop_normal1d_log], [sample_prop_normal1d], [0.5,0.5],
    #                                                niter=niter, logpdf=true)
    # @show accept_ratio
    # test_mean_std(mvn, sa2_mh_cw_log)

    # ## plot
    # figure(); hold(true)
    # plothist2(sa2, bins, "built-in")
    # plothist2(sa2_rej, bins, "rejection (uniform)")
    # plothist2(sa2_rej2, bins, "rejection (id)")
    # plothist2(sa2_m, bins, "metro")
    # plothist2(sa2_mh, bins, "metro-hast")
    # legend()

    ##fit

    # @show fit(MvNormal, sa2_rej2)
    # @show fit(MvNormal, sa2_m)
    # @show fit(MvNormal, sa2_m_cw)
    # @show fit(MvNormal, sa2_mh)
    # @show fit(MvNormal, sa2_mh_cw)

elseif dimensions==10

end

## Rosenbrock

"Rosenbrock log pdf"
logpdf = @anon (x,y) -> -( 100*(y-x^2)^2+(1-x)^2 )/20

# samples = emcee(data, fwd, likelihood!, sample_likelihood, prior, theta0;
#                nchains=10^2,
#                niter=10^4,
#                nburnin=round(Int,niter/10),
#                nthin=1,
#                logpdf=true,
#                datamask=nothing, # if set to a bool array
#                a_scale=2.0 # step scale parameter.  Probably needn't be adjusted
#                )


nothing
