using CheckerboardAMDGPU
using AMDGPU

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
	neighbor_table = [1 3 5 7 9 11 13 15 4 2 8 6 12 10 16 14 1 2 3 4 9 10 11 12 13 14 15 16 5 6 7 8;
		  2 4 6 8 10 12 14 16 1 3 5 7 9 11 13 15 5 6 7 8 13 14 15 16 1 2 3 4 9 10 11 12]

	coshΔτt = fill(cosh(t*Δτ), size(neighbor_table,2))
	sinhΔτt = fill(sinh(t*Δτ), size(neighbor_table,2))
	colors = Matrix{Int32}(
		     [ 1  9 17 25;
			   8 16 24 32])

    # calculate exact exponentiated hopping matrix exp(-Δτ⋅K)
	K = zeros(Float64, N, N)
    for c in axes(neighbor_table,2)
		# Note: exp([[0,t], [t*, 0]]) = [[c, s], [s*, c]]
		# where c = cosh(|t|), s = sinh(|t|) t/|t|
        i      = neighbor_table[1,c]
        j      = neighbor_table[2,c]
        K[j,i] = -t
        K[i,j] = conj(-t)
    end
    expnΔτK = Hermitian(exp(-Δτ*K))

	nt = AMDGPU.ROCArray{Int32,2}(neighbor_table)
	ch = AMDGPU.ROCArray{Float64,1}(coshΔτt)
	sh = AMDGPU.ROCArray{Float64,1}(sinhΔτt)

#   # build dense versions of matrices
    I_dense   = Matrix{Float64}(I,L^2,L^2)
    Γ_dense   = AMDGPU.ROCMatrix{Float64}(I_dense)
    Γᵀ_dense  = AMDGPU.ROCMatrix{Float64}(I_dense)
    Γ⁻¹_dense = AMDGPU.ROCMatrix{Float64}(I_dense)
	CheckerboardAMDGPU.checkerboard_rmul!(Γ_dense,  nt, ch, sh,
										  colors; transposed=false, inverted=false)
    CheckerboardAMDGPU.checkerboard_rmul!(Γᵀ_dense, nt, ch, sh,
										  colors; transposed=true, inverted=false)
    CheckerboardAMDGPU.checkerboard_rmul!(Γ⁻¹_dense,nt, ch, sh,
										  colors; transposed=false, inverted=true)

	A   = Matrix{Float64}(undef,L^2,L^2)
	copyto!(A, Γ_dense)
	@show view(A, 1:3, 1:3)
    @test norm(A-expnΔτK)/norm(expnΔτK) < 0.02001

	Aᵀ  = Matrix{Float64}(undef,L^2,L^2)
	copyto!(Aᵀ, Γᵀ_dense)
	@show view(Aᵀ, 1:3, 1:3)
	@test norm(A - transpose(Aᵀ)) < 1e-6

	A⁻¹ = Matrix{Float64}(undef,L^2,L^2)
	copyto!(A⁻¹, Γ⁻¹_dense)

	result = Matrix{Float64}(undef,L^2,L^2)
	mul!(result, A⁻¹, A)
	@test norm(I_dense - result) < 1e-6
	#@test A⁻¹ ≈ inv(A)
end
