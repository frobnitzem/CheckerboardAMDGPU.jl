using CheckerboardAMDGPU
using Checkerboard
using LinearAlgebra
using Test

@testset "CheckerboardAMDGPU.jl" begin
    
    # construct reference results
    L  = 4 # lattice size
	N = L^2
	t  = 1.0
	Δτ = 0.1
    #Γ = CheckerboardMatrix(nt, t, Δτ)
	nt = [1 3 5 7 9 11 13 15 4 2 8 6 12 10 16 14 1 2 3 4 9 10 11 12 13 14 15 16 5 6 7 8;
		  2 4 6 8 10 12 14 16 1 3 5 7 9 11 13 15 5 6 7 8 13 14 15 16 1 2 3 4 9 10 11 12]
	coshΔτt = fill(cosh(t*Δτ), size(nt,2))
	sinhΔτt = fill(sinh(t*Δτ), size(nt,2))
	colors = [1 9 17 25;
			  8 16 24 32]

    # calculate exact exponentiated hopping matrix exp(-Δτ⋅K)
	K = zeros(Float64, N, N)
    for c in axes(nt,2)
        i      = nt[1,c]
        j      = nt[2,c]
        K[j,i] = -t
        K[i,j] = conj(-t)
    end
    expnΔτK = Hermitian(exp(-Δτ*K))

    # test vector multiplication
    v  = randn(N)
    v′ = copy(v)
	checkerboard_lmul!(v′, nt, coshΔτt, sinhΔτt, colors)
	
    u = zeros(N)
	mul!(u, expnΔτK, v)
	@test norm(v′-u)/norm(u) < Δτ*0.5

	checkerboard_lmul!(v′, nt, coshΔτt, sinhΔτt, colors; inverted=true)
    @test v′ ≈ v

#   # build dense versions of matrices
#   I_dense   = Matrix{Float64}(I,L^2,L^2)
#   Γ_dense   = similar(I_dense)
#   Γᵀ_dense  = similar(I_dense)
#   Γ⁻¹_dense = similar(I_dense)
#   mul!(Γ_dense,   Γ,   I_dense)
#   mul!(Γᵀ_dense,  Γᵀ,  I_dense)
#   mul!(Γ⁻¹_dense, Γ⁻¹, I_dense)

#   @test Γ_dense ≈ transpose(Γᵀ_dense)
#   @test Γ_dense ≈ inv(Γ⁻¹_dense)
#   @test norm(Γ_dense-expnΔτK)/norm(expnΔτK) < 0.01

#   A = similar(Γ_dense)
#   B = similar(Γ_dense)
#   C = similar(Γ_dense)

#   mul!(A,Γ,I_dense)
#   mul!(B,I_dense,Γ)
#   @test A ≈ B

#   @. C = rand()
#   mul!(A,Γ,C)
#   mul!(B,C,Γ)
#   @test !(A ≈ B)
end
