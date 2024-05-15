module Samplers
using Random

export GaussianSampler, MetropolisHastingsSampler

include("types.jl")
include("gaussian_sampler.jl")
include("metropolis_hastings_sampler.jl")
end # end of 
