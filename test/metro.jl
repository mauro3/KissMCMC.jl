# test Metropolis
@testset "metropolis" begin
    for tc in testcases
        # @show tc[:dist]
        pdf = make_lpdf(tc.dist)
        thetas, accept_ratio, logdensities, blobs =
            metropolis(pdf, tc.sample_ppdf, tc.theta0;
                       niter=tc.niter,
                       hasblob=tc.hasblob,
                       use_progress_meter=false);
        @test length(thetas)==tc.niter÷2
        @test 0.15<accept_ratio<0.45
        test_mean_std(thetas, tc, tc.tolm)
        #test_blobs(tc.truths, blobs)
        #test_logdensities(tc.truths, logdensities, pdf)
    end
end
