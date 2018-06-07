if VERSION>=v"0.5-"
    addprocs(2) # segfaults on 0.4
end
@everywhere using KissMCMC
@everywhere using Base.Test, Compat

@everywhere import Distributions


# helpers
function test_mean_std(s1,s2,diff=0.02)
    err_mean = abs(mean(s1)-mean(s2)) / max(abs(mean(s1)),0.01)
    err_std = abs(std(s1)-std(s2)) / max(abs(std(s1)),0.01)
    if err_mean > diff
        print("--->  ")
        warn("Mean is off my more than $(diff*100)%! err_mean=$err_mean")
    end
    if err_std > diff
        print("--->  ")
        warn("Std is off my more than $(diff*100)%! err_std=$err_std")
    end
end
function test_mean_std(s1::Matrix,s2::Matrix, diff=0.15)
    for i=1:size(s1,1)
        err_mean = abs(mean(s1[i,:])-mean(s2[i,:])) / max(abs(mean(s1[i,:])),0.01)
        err_std = abs(std(s1[i,:])-std(s2[i,:])) / max(abs(std(s1[i,:])),0.01)
        if err_mean > diff
            print("--->  ")
            warn("Mean is off my more than $(diff*100)%! err_mean=$err_mean in row $i")
        end
        if err_std > diff
            print("--->  ")
            warn("Std is off my more than $(diff*100)%! err_std=$err_std in row $i")
        end
    end
end

# Without this it gives strange errors:
# TODO file bug report
logpdf = 1
builtin = 1
pdf = 1

# number of samples to draw
@everywhere n = Int(10^5)
@everywhere nchains = 50

# tests
for dimensions=1:2
    if dimensions==1
        println("One dimensional tests:\n")

        # our mystery (unscaled) distribution:
        rand_025() = rand()-0.25
        rand_025(n) = typeof(rand_025())[rand_025() for i=1:n]

        randn_1() = randn()+1
        randn_1(n) = randn(n)+1

        # log-pdfs
        unif =  x->-0.25<=x<=0.75 ? 0.0:-Inf
        npdf =  x ->  -(x-1)^2/2
        exppdf =  x -> x<0 ? -Inf : -x

        tests = [(rand_025  , unif, -0.5:0.01:1, "Uniform"),
                 (randn_1, npdf , -5:0.1:5, "Normal"), # normal, bins
                 (randexp, exppdf, 0:0.1:10, "Exponential")]

        for test in tests
            builtin, logpdf, bins, name = test # need const to make nested closures fast
            pdf =  x->exp(logpdf(x))
            println("\nTesting $name distribution:")

            # samples drawn with built-in sampler
            print("Built-in sampler: ")
            builtin(1)
            @time sa = builtin(n)

            # rejection sampling
            print("Uniform rejection sampler: ")
            @time sa_rej = Float64[rejection_sample_unif(pdf, [minimum(bins),maximum(bins)], 1) for i=1:n]
            @show typeof(sa), typeof(sa_rej)
            test_mean_std(sa, sa_rej)

            # rejection sampling v2
            print("General rejection sampler: ")
            f =  x->2*pdf(x) # proposal distribution
            g =  () -> builtin()
            @time sa_rej2 = Float64[rejection_sample(pdf, f, g) for i=1:n]

            test_mean_std(sa, sa_rej2)

            # # MCMC
            # sample PDF

            const sigma = 2.0
            sample_prop_normal =  x->randn()*sigma+x

            println("Serial MCMC")
            # # warmup
            metropolis(logpdf, sample_prop_normal, 0.5, niter=10, logpdf=true)
            print("Metropolis     : ")
            @time thetas, accept_ratio, blobs, logposts = metropolis(logpdf, sample_prop_normal, 0.5, niter=n)
            test_mean_std(sa, thetas)
            @test blobs==nothing
            @test length(logposts)==length(thetas)

            emcee(logpdf, (0.5, 0.1), niter=100, nchains=10);
            print("emcee           : ")
            @time thetas_e, accept_ratio_e, blobs, logposts_e = emcee(logpdf, (0.5, 0.1), niter=n, nchains=100);
            @test blobs==nothing

            thetas_e_, accept_ratio_e_, blobs, logposts_e_ = squash_chains(thetas_e, accept_ratio_e, blobs, logposts_e );
            @test blobs==nothing
            @test length(logposts_e_)==length(thetas_e_)

            test_mean_std(sa, thetas_e_)
            thetas_e_, accept_ratio_e_ = squash_chains(thetas_e, accept_ratio_e, order=true );
            test_mean_std(sa, thetas_e_)
        end

    elseif dimensions==2
        println("\n\nTwo dimensional tests:\n")
        #### Multivariate distributions

        @everywhere begin
            # narrow Normal
            const mu = [0.5,-0.25]
            const covm = [0.47 1.8
                          1.8 7.]
            const icovm = inv(covm)
            mvn = Distributions.MvNormal(mu, covm)

            #        immutable NormPDF2 end
            #        call(::NormPDF2, x) =
            npdf2 =  x-> (-(x-mu)'*icovm*(x-mu)/2)[1]

            # Rosenbrock
            rosenpdf =  x-> -( 100*(x[2]-x[1]^2)^2+(1-x[1])^2 )/20

            tests2 = Vector{Any}[Any[
                                    (() -> rand(mvn)),
                                    ((n) -> rand(mvn,n)),
                                    npdf2,
                                    StepRangeLen{Float64}[-3:0.1:4, -12:0.1:12],
                                    "Normal",
                                    0.15
                                    ],
                                Any[
                                    (() -> NaN), # no analytic available
                                    ((n) -> NaN*ones(n)),
                                    rosenpdf,
                                    StepRangeLen{Float64}[-4:0.1:6, -1:0.1:31],
                                    "Rosenbrock",
                                    0.3
                                    ]
                                ]
            pdf =  x->exp(logpdf(x))

            const sigma = 2.0
            sample_prop_normal2 =  x0 ->randn(2)*sigma+x0

        end
        # make a "truth" for Rosenbrock
        print("Calculating Rosenbrock standard...")
        emcee(rosenpdf, ([0.5,0.5], 0.1), niter=100, nchains=10);
        thetas_e, accept_ratio_e = emcee(rosenpdf, ([1.0,1.0], 0.1), niter=n, nchains=100);
        thetas_rosen, accept_ratio_e = squash_chains(thetas_e, accept_ratio_e );
        thetas_rosen = thetas_rosen[:,1:1000:end]
        println("done.")

        for test in tests2
            builtin, builtin_n, logpdf, bins, name, diff = test
            println("\nTesting $name distribution:")

            # samples drawn with built-in sampler
            if logpdf==npdf2
                print("Built-in sampler: ")
                builtin_n(1)
                @time sa2 = builtin_n(n)
            else
                sa2 = thetas_rosen
            end

            # rejection sampling
            print("Uniform rejection sampler using $(n/100): ")
            rejection_sample_unif(pdf, Vector{Float64}[map(minimum, bins),map(maximum, bins)], 1.0)
            @time sa2_rej = hcat(Vector{Float64}[rejection_sample_unif(pdf, Vector{Float64}[map(minimum, bins),map(maximum, bins)], 1.0) for i=1:n÷100]...)
            test_mean_std(sa2, sa2_rej, diff)
            if logpdf==npdf2
                f =  x->2*pdf(x)
                g =  () -> builtin()
                print("General rejection sampler using $(n/10): ")
                rejection_sample(pdf, f, g)
                @time sa2_rej2 = hcat(Vector{Float64}[rejection_sample(pdf, f, g) for i=1:n÷100]...)
                test_mean_std(sa2, sa2_rej2, diff)
            else
                sa2 = thetas_rosen
            end

            # # MCMC
            # sample PDF

            println("Serial MCMC")
            # # warmup
            metropolis(logpdf, sample_prop_normal2,  [0.5, 0.5], niter=10, logpdf=true)
            print("Metropolis     : ")
            @time thetas, accept_ratio = metropolis(logpdf, sample_prop_normal2, [0.5, 0.5], niter=n)
            test_mean_std(sa2, thetas, diff)

            emcee(logpdf, ([0.5,0.5], 0.1), niter=100, nchains=10);
            print("emcee           : ")
            @time thetas_e, accept_ratio_e = emcee(logpdf, ([0.5,0.5], 0.1), niter=n, nchains=100);
            thetas_e_, accept_ratio_e_ = squash_chains(thetas_e, accept_ratio_e );
            test_mean_std(sa2, thetas_e_, diff)
            thetas_e_, accept_ratio_e_ = squash_chains(thetas_e, accept_ratio_e, order=true);
            test_mean_std(sa2, thetas_e_, diff)
        end

        println("\n\nTwo dimensional parallel tests:\n")
        # note that these are slower than the serial one.  But for expensive logpdf they maybe faster.
        if VERSION<v"0.5-"
            println("skipping because of segfaults.")
        else
            for test in tests2
                builtin, builtin_n, logpdf, bins, name, diff = test
                println("\nTesting $name distribution:")

                if logpdf==npdf2
                    print("Built-in sampler: ")
                    builtin_n(1)
                    @time sa2 = builtin_n(n)
                else
                    sa2 = thetas_rosen
                end

                ## MCMC parallel
                n_workers = max(length(procs())-1,1)
                println("Parallel MCMC; number of workers: $n_workers")

                # make one chain per worker
                a = n_workers>1 ? linspace(0.1,0.9, n_workers):nothing
                theta0 = n_workers==1 ?  Vector{Float64}[[0.5, 0.5]] : [[0.5,a[i]] for i=1:n_workers]

                print("Metropolisp    : ")
                metropolisp(logpdf, sample_prop_normal2, theta0, niter=10, logpdf=true)
                @time thetasp, accept_ratiop = metropolisp(logpdf, sample_prop_normal2, theta0, niter=n÷n_workers)
                thetasp_, accept_ratiop_ = squash_chains(thetasp, accept_ratiop );
                test_mean_std(sa2, thetasp_, diff)
                thetasp_, accept_ratiop_ = squash_chains(thetasp, accept_ratiop, order=true );
                test_mean_std(sa2, thetasp_, diff)

                print("emceep          : ")
                emceep(logpdf, ([0.5, 0.5], 0.1), niter=100, nchains=10);
                @time thetas_ep, accept_ratio_ep = emceep(logpdf, ([0.5,0.5], 0.1), niter=n÷5, nchains=nchains);
                thetas_ep_, accept_ratio_ep_ = squash_chains(thetas_ep, accept_ratio_ep );
                test_mean_std(sa2, thetas_ep_, diff)
                thetas_ep_, accept_ratio_ep_ = squash_chains(thetas_ep, accept_ratio_ep, order=true );
                test_mean_std(sa2, thetas_ep_, diff)
            end
        end
    end
end
