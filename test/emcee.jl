# test g-distribution
@testset "g-dist" begin
    a = 3.5
    samples2 = [KissMCMC.sample_g(a) for i=1:50000];

    @test all(1/a .<= samples2 .<= a)
    @test KissMCMC.cdf_g_inv(1, a) ≈ a
    @test KissMCMC.cdf_g_inv(0, a) ≈ 1/a
    z = 1/a:0.01:a
    meang = sum(z.*KissMCMC.g_pdf.(z, a)) *step(z)
    @test isapprox(mean(samples2), meang, atol=1e-2)
    stdg = sqrt(sum((meang.-z).^2 .* KissMCMC.g_pdf.(z, a)) *step(z))
    @test isapprox(std(samples2), stdg, atol=1e-2)
end

# test MCMC
@testset "emcee" begin
    for tc in testcases
        pdf = make_lpdf(tc.dist)
        @inferred pdf(tc.theta0)
        theta0s = make_theta0s(tc.theta0, tc.ball_radius,
                               pdf, tc.nwalkers,
                               hasblob=tc.hasblob)
        samples = emcee(pdf, theta0s;
                        niter=tc.niter,
                        hasblob=tc.hasblob,
                        use_progress_meter=false,
                        tc.otherkws...);
        @test length.(samples[1:3]) == (tc.nwalkers, tc.nwalkers, tc.nwalkers)
        if tc.hasblob
            @test length(samples[4]) == tc.nwalkers
        else
            @test samples[4]==nothing
        end
        @test length(samples[1][1]) == tc.niter÷tc.nwalkers÷2 # ÷2 because of burnin
        thetas, accept_ratio, logdensities, blobs = squash_walkers(samples...;
                                                                   verbose=false,
                                                                   tc.squash_walkers_kws...)

        !tc.hasblob && @test blobs==nothing
        @test length(thetas) == tc.niter÷2
        @test length(logdensities) == tc.niter÷2
        @test accept_ratio>0.1
        test_mean_std(thetas, tc, tc.tole)
        test_blobs(tc.blob_truths, blobs)
        #test_logdensities(tc.truths, logdensities, pdf)
    end
end
