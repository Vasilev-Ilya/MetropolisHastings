"""
    SimpleIntegrator(; 
        func, 
        sampels_density, 
        bounds, 
        N=100_000, 
        init_state::Float64=1.0, 
        proposal_std::Float64=1.0, 
    )

"""
function SimpleIntegrator(; 
    func, 
    sampels_density, 
    bounds, 
    N=100_000, 
    init_state::Float64=1.0, 
    proposal_std::Float64=1.0, 
)
    sampels = SimpleMHS(
        N, 
        sampled_dist=sampels_density, 
        init_state=init_state, 
        proposal_std=proposal_std, 
        bounds=bounds, 
    )
    mapped_sampels = func.(sampels)
    return (bounds[2] - bounds[1]) * sum(mapped_sampels) / N
end
