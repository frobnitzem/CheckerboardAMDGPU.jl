using Checkerboard
using LatticeUtilities

using AMDGPU
using CheckerboardAMDGPU

# size of square lattice
function setup(L)
	# define square lattice unit cell
	square  = UnitCell(
		lattice_vecs = [[1.,0.],[0.,1.]],
		basis_vecs = [[0.,0.]]
	)

	# define L×L unit cell lattice with periodic boundary conditions
	lattice = Lattice(
		L = [L,L],
		periodic = [true,true]
	)

	# define neasrest-neighbor bond in x direction
	bond_x  = Bond(
		orbitals = (1,1),
		displacement = [1,0]
	)

	# define nearest-neighbor bond in y direction
	bond_y  = Bond(
		orbitals = (1,1),
		displacement = [0,1]
	)

	# get the number of size in the lattice, i.e. N = L×L for square lattice
	N = nsites(square, lattice)

	# build the neighbor table for an L×L square lattice with periodic boundary condtions
	# and just nearest-neighbor bonds
	neighbor_table = build_neighbor_table([bond_x,bond_y], square, lattice)

	# define uniform hopping amplitude/energy for corresponding square lattice tight-binding model
	t = ones(size(neighbor_table,2))

	# define discretization in imaginary time i.e. the small parameter the in checkerboard approximation
	Δτ = 0.1

	# construct/calculate checkerboard approximation
	return N, CheckerboardMatrix(neighbor_table, t, Δτ)
end

function julia_main(args::Array{String,1})::Cint
	# must initialize scalars
    L::Int32 = 16
    steps::Int32 = 1

    @show args

    # args don't include Julia executable and program
    nargs = size(args)[1]

    if nargs == 2
		L = parse(Int32, args[1])
        steps = parse(Int32, args[2])
    else
        throw( ArgumentError(string("Usage: example.jl <L> <steps>")) )
    end
	if steps < 1
		throw( ArgumentError("Steps must be at least 2") )
	end

	(N, (; neighbor_table, coshΔτt, sinhΔτt, colors, transposed, inverted)) = setup(L)

	# define M random vectors of length N
	M = 5
	print("Time to allocate X")
	@time X = AMDGPU.ROCArray{Float32,2}(undef, M, N)
    print("Time to fill X")
    @time AMDGPU.rand!(X)

	timings = zeros(steps)
	for i = 1:steps
		timings[i] = @elapsed CheckerboardAMDGPU.checkerboard_rmul!(
							X, neighbor_table, coshΔτt, sinhΔτt, colors,
							transposed = transposed, inverted = inverted)
	end
	average_time = sum(timings[2:steps]) / (M*(steps - 1))
	println("Size ", M, "x", N, ", time per M per step (us) ", average_time*1e6)

	return 0
end

julia_main(ARGS)
