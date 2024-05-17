module Integrators

using Samplers

export SimpleIntegrator, NormalDensity

include("densities.jl")
include("simple_integrator.jl")

end # module Integrators
