# test/runtests.jl
using Test
using LinearAlgebra 
using ..SerialBiCGSTAB 

@testset "Test 1: 2x2 system" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    tol = 1e-8
    max_iter = 100

    state = initialize_state(A, b, tol=tol, max_iter=max_iter)
    @time x, iters = solve_bicgstab!(state)
    println(x, iters)
    x_expected = A \ b
    println(x_expected)

    @test isapprox(x, x_expected; atol=tol*10) 
    @test iters <= max_iter 
    @test norm(A*x - b) <= tol 
end

@testset "Test 2: ((initial guess) = solution)" begin
    A = Diagonal([2.0, 5.0]) #sparse matrix 😱
    b = [2.0, 5.0]
    x_exact = [1.0, 1.0]
    tol = 1e-8

    state = initialize_state(A, b, tol=tol, max_iter=10)
    state.x .= x_exact
    state.r .= b - A * state.x
    @test norm(state.r) <= tol 

    x, iters = solve_bicgstab!(state)

    @test iters == 0
    @test x == x_exact
    @test norm(A * x - b) <= tol
end

@testset "Test 3: rand pos-symmetric-def system" begin
    n = 5
    A_rand = rand(n, n)
    A = A_rand * A_rand' + n * I #stolen idk how this works
    x_exact = ones(n)
    b = A * x_exact
    tol = 1e-7
    max_iter = 50

    state = initialize_state(A, b, tol=tol, max_iter=max_iter)
    @time x, iters = solve_bicgstab!(state)

    @test iters <= max_iter
    @test isapprox(x, x_exact, rtol=tol*100, atol=tol*10)
    initial_residual_norm = norm(b - A*zeros(n)) 
    @test norm(A * x - b) < tol * max(1.0, initial_residual_norm) #TODO
    @test norm(state.r) < tol * max(1.0, initial_residual_norm) 
end

@testset "Test 4: rho_new ≈ 0" begin
    A = [1.0 0; 0 1]
    b = [1.0, 1.0]
    state = initialize_state(A, b)

    state.r_hat = [-1.0, 1.0] 
    @test isapprox(dot(state.r_hat, state.r), 0.0, atol=1e-15) #r=b=[1,1] * [-1,1] = 0 
    @test_throws ErrorException("rho_new == 0") SerialBiCGSTAB.step!(state, 1) #TODO change error msg 
end

@testset "Test 5: a 3x3 system" begin
    A = [3.0 2.0 -1.0;
         2.0 -2.0 4.0;
        -1.0 0.5 -1.0]
    b = [1.0, -2.0, 0.0]
    tol = 1e-8
    max_iter = 30

    state = initialize_state(A, b, tol=tol, max_iter=max_iter)
    @time x, iters = solve_bicgstab!(state)
    x_expected = A \ b

    @test isapprox(x, x_expected; atol=tol*10)
    @test iters < max_iter
    @test norm(A*x - b) < tol
end