# test Metropolis
@testset "metropolis" begin
    for tc in testcases
        @show tc.dist
        pdf = make_lpdf(tc.dist)
        @inferred pdf(tc.theta0)
        thetas, accept_ratio, logdensities, blobs =
            metropolis(pdf, tc.sample_ppdf, tc.theta0;
                       niter=tc.niter,
                       hasblob=tc.hasblob,
                       use_progress_meter=false,
                       tc.otherkws...);
        !tc.hasblob && @test blobs==nothing
        @test length(thetas)==tc.niter÷2
        @test length(logdensities)==tc.niter÷2
        @test 0.15<accept_ratio<0.45
        test_mean_std(thetas, tc, tc.tolm)
        test_blobs(tc.blob_truths, blobs)
        #test_logdensities(tc.truths, logdensities, pdf)
    end
end
