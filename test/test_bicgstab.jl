using Test
using ..SerialBiCGSTAB

@testset "Basic 2x2 BiCGSTAB" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    tol = 1e-8
    max_iter = 100

    state = initialize_state(A, b,tol=tol, max_iter=max_iter)
    @time x, iters = solve_bicgstab!(state)
    x_expected = A \ b 

    @test isapprox(x, x_expected; atol=1e-6)
    @test iters < max_iter
end

@testset "Basic 3x3 BiCGSTAB" begin
    A = [3.0 2.0 -1.0;
         2.0 -2.0 4.0;
        -1.0 0.5 -1.0]
    b = [1.0, -2.0, 0.0]
    tol = 1e-8
    max_iter = 20

    state = initialize_state(A, b,tol=tol, max_iter=max_iter)
    @time x, iters = solve_bicgstab!(state)
    x_expected = A \ b 

    @test isapprox(x, x_expected; atol=1e-6)
    @test iters < max_iter
end



#@time ig this shows allocation as well 
#We want to check how much memory we have allocated
#solve phase something ?? bruh 

