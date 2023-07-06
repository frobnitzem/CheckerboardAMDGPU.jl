# CheckerboardAMDGPU

A version of [Checkerboard.jl](https://github.com/cohensbw/Checkerboard.jl) created
for AMDGPU.

GPU kernel calling is based on this [simple-gemm example](https://github.com/williamfgc/simple-gemm/tree/main/julia/GemmDenseAMDGPU/src).

Benchmarking shows that the time taken to run the kernel is
dominated by the kernel launch latency:

```
$ for n in 16 32 64 128; do julia -O2 --project=. example.jl $n 128 1000; done
Size 128x256, time per M per step (us) 5.8708339589589595
Size 128x1024, time per M per step (us) 5.999556650400401
Size 128x4096, time per M per step (us) 6.348324832645146
Size 128x16384, time per M per step (us) 7.90716812124625

$ for n in 16 32 64 128; do julia -O2 --project=. example.jl $n 64 1000; done
Size 64x256, time per M per step (us) 11.6896200731982
Size 64x1024, time per M per step (us) 11.771267282907909
Size 64x4096, time per M per step (us) 12.628145442317313
Size 64x16384, time per M per step (us) 14.428985047547549
```
