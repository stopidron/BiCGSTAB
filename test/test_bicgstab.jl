using Test
using ..SerialBiCGSTAB

@testset "Basic 2x2 BiCGSTAB" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    tol = 1e-8
    max_iter = 100

    state = initialize_state(A, b,tol=tol, max_iter=max_iter)
    x, iters = solve_bicgstab!(state)
    x_expected = A \ b

    @test isapprox(x, x_expected; atol=1e-6)
    @test iters < max_iter
end
