using Bornsampling
import FiniteMPS
using LinearAlgebra
using Random
using Test

const TK = FiniteMPS.TensorKit
const BS = Bornsampling

Random.seed!(0x5eed)

include("helpers.jl")

@testset "Bornsampling.jl" begin
    include("contraction.jl")
    include("residual.jl")
    include("localspaces.jl")
    include("factor.jl")
    include("all_factors.jl")
    include("probability.jl")
    include("joint.jl")
    include("api.jl")
    include("batch.jl")
    include("performance.jl")
end
