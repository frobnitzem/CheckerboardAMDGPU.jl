####################################
## MATRIX-MATRIX MULTIPLY METHODS ##
####################################

Continuous = Union{AbstractFloat, Complex{<:AbstractFloat}}

#function color_rmul_kernel!(A::AMDGPU.Device.ROCDeviceMatrix{Float32, 1},
#					start::Int32, stop::Int32,
#					nt::AMDGPU.Device.ROCDeviceMatrix{Int32, 1},
#					ch::AMDGPU.Device.ROCDeviceVector{Float64, 1},
#					sh::AMDGPU.Device.ROCDeviceVector{Float64, 1},
#					inverse::Int32)
function color_rmul_kernel!(A, start, stop, nt, ch, sh, inverse)
    i0 = AMDGPU.blockIdx().x+start-1
	Ni = AMDGPU.gridGroupDim().x # gridDim
        
    k0 = AMDGPU.threadIdx().x
    Nk = AMDGPU.blockDim().x

	# causes terrible compiler error:
	#for n = i0:Ni:stop
	n = i0
	while n <= stop
		i = nt[1,n]
		j = nt[2,n]
		cij = ch[n]
		sij = inverse * sh[n]
		#for k = k0:Nk:size(A,1)
		k = k0
		while k <= size(A,1)
			vi = A[k,i]
			vj = A[k,j]
			vi = cij*vi + conj(sij)*vj
			vj = sij*vi + cij*vj
			A[k,i] = vi
			A[k,j] = vj
			k += Nk
		end
		n += Ni
    end

    return
end

function color_rmul!(A::AMDGPU.ROCMatrix{T},
		start::Int32, stop::Int32,
		neighbor_table::AMDGPU.ROCMatrix{Int32},
        coshΔτt::AMDGPU.ROCVector{E}, sinhΔτt::AMDGPU.ROCVector{E};
		inverted::Bool = false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by complex a checkerboard matrix!"

	#https://amdgpu.juliagpu.org/stable/kernel_launch/
	#kernel = @roc launch=false mykernel(args...)
	#@show AMDGPU.Compiler.calculate_occupancy(kernel.fun, AMDGPU.default_device())

	# q = AMDGPU.ROCQueue()
	# queue = q
	inverse::Int32 = 1 - 2*inverted

	blocks = min(220, stop-start+1)
	threads = 64
	#println("Starting kernel ", start, ":", stop)
	#@show size(A)
	#@show size(neighbor_table)
	#@show size(coshΔτt)
	#@show size(sinhΔτt)
	return AMDGPU.@roc threads = threads blocks = blocks color_rmul_kernel!(
								A, start, stop, neighbor_table,
								coshΔτt, sinhΔτt, inverse)
end

function checkerboard_rmul!(A::AMDGPU.ROCMatrix{T},
		neighbor_table::AMDGPU.ROCMatrix{Int32},
        coshΔτt::AMDGPU.ROCVector{E}, sinhΔτt::AMDGPU.ROCVector{E},
        colors::AbstractMatrix{Int32};
		transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by complex a checkerboard matrix!"

	# number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

	event = nothing
	for color in start:step:stop
		event = color_rmul!(A, colors[1,color], colors[2,color],
							neighbor_table, coshΔτt, sinhΔτt, inverted=inverted)
	end
	# Events run in-order, so it suffices to wait for the last one.
	if event != nothing
		AMDGPU.wait(event)
	end

	return nothing
end
