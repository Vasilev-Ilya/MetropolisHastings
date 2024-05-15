function MetropolisHastingsSampler(
    N::Int=1; 
    sampled_dist,
    initial_state::Float64=0.0, 
    proposal_std::Float64=1.0, 
)::Sampels
    sampels = zeros(N)
    curr_state = initial_state
    for i=1:N
        new_state = GaussianSampler(mu=curr_state, sigma=proposal_std)[1]
        acceptance_ratio = min(
            1.0, 
            sampled_dist(new_state) / sampled_dist(curr_state),
        )
        accept_prob = rand()
        sampels[i] = if accept_prob <= acceptance_ratio
            new_state
        else
            curr_state
        end
        curr_state = sampels[i]
    end

    return sampels
end