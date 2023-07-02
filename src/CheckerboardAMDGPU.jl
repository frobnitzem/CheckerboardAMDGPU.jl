module CheckerboardAMDGPU

import AMDGPU

# low-level routines for in-place matrix-vector and matrix-matrix products
include("matrix_matrix_mul.jl")
#include("matrix_vector_mul.jl")
export checkerboard_rmul!
export checkerboard_color_rmul!

end #module
