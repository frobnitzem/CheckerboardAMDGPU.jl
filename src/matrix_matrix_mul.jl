####################################
## MATRIX-MATRIX MULTIPLY METHODS ##
####################################

Continuous = Union{AbstractFloat, Complex{<:AbstractFloat}}

function gemm!(A, B, C)
    row =
        (AMDGPU.workgroupIdx().x - 1) * AMDGPU.workgroupDim().x +
        AMDGPU.workitemIdx().x
    col =
        (AMDGPU.workgroupIdx().y - 1) * AMDGPU.workgroupDim().y +
        AMDGPU.workitemIdx().y

    sum = Float32(0.0)

    if row <= size(A, 1) && col <= size(B, 2)

        for i = 1:size(A, 2)
            @inbounds sum += A[row, i] * B[i, col]
        end
        C[row, col] = sum
    end

    return
end

function callGemm(A::AMDGPU.ROCArray{Float32,2}, B::AMDGPU.ROCArray{Float32,2}, C::AMDGPU.ROCArray{Float32,2})

		BLOCK_SIZE = 32
        # Julia is column-based (like Fortran)
        #print("Time to allocate A")
        #@time A = AMDGPU.ROCArray{Float32,2}(undef, A_rows, A_cols)

        #print("Time to allocate B")
        #@time B = AMDGPU.ROCArray{Float32,2}(undef, B_rows, B_cols)

        #print("Time to initialize C")
        #@time C = AMDGPU.zeros(Float32, A_rows, B_cols)

        #print("Time to fill A")
        #@time AMDGPU.rand!(A)
        #print("Time to fill B")
        #@time AMDGPU.rand!(B)

		grid = (size(A,1), size(B,2))
        threads = (BLOCK_SIZE, BLOCK_SIZE)
		event = AMDGPU.@roc groupsize = threads gridsize = grid gemm!(A, B, C)
		AMDGPU.wait(event)

    return nothing
end

function checkerboard_rmul!(A::AMDGPU.ROCArray{T,2}, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::Matrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by complex a checkerboard matrix!"
	return nothing
end
