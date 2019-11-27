# test g-distribution
a = 3.5
samples1 = metropolis(x->KissMCMC.g_pdf(x, a), x->0.1*randn(), 1.0)[1];
samples2 = [KissMCMC.sample_g(a) for i=1:50000];

@test all(1/a .<= samples2 .<= a)
@test KissMCMC.cdf_g_inv(1, a) ≈ a
@test KissMCMC.cdf_g_inv(0, a) ≈ 1/a
z = 1/a:0.01:a
meang = sum(z.*KissMCMC.g_pdf.(z, a)) *step(z)
@test isapprox(mean(samples2), meang, atol=1e-2)
stdg = sqrt(sum((meang.-z).^2 .* KissMCMC.g_pdf.(z, a)) *step(z))
@test isapprox(std(samples2), stdg, atol=1e-2)

#
