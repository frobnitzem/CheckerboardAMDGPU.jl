import CheckerboardAMDGPU

function julia_main(args)::Cint
    CheckerboardAMDGPU.main(args)
    return 0
end

julia_main(ARGS)
