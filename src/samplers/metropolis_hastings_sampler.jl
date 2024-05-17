# Реализация одномерного и многомерного сэплеров, использующих в качестве алгоритма
# сэмплирования классический Metropolis–Hastings Algorithm с гауссовским proposal 
# (https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm).

"""
    SimpleMHS(
        N::Int=1; 
        sampled_dist,
        proposal_std::Float64=1.0, 
    )

 Одномерный Metropolis–Hastings сэмплер.

Arguments
≡≡≡≡≡≡≡≡≡

    - N: число сэмплируемых псевдослучайных чисел;

    - sampled_dist: функция сэмплируемого распределения;

    - init_state: начальное значение сэмпла;

    - proposal_std: среднеквадратичное отклонение функция плотности proposal;

    - bounds: границы значения сэмплов.

Returns
≡≡≡≡≡≡≡

    -  Список сэмплируемых значений типа Float64.
"""
function SimpleMHS(
    N::Int=1; 
    sampled_dist, 
    init_state::Float64=1.0, 
    proposal_std::Float64=1.0, 
    bounds::Tuple{Float64, Float64}=(-Inf, Inf)
)::Matrix_f64
    sampels = zeros(N, 1)
    curr_state = init_state
    accept_prob = rand(N)
    for i=1:N
        new_state = SimpleGS(mu=curr_state, sigma=proposal_std)[1]
        if new_state > bounds[2]
            new_state = bounds[2]
        elseif new_state < bounds[1]
            new_state = bounds[1]
        end
        acceptance_ratio = min(
            1.0, 
            sampled_dist(new_state) / sampled_dist(curr_state),
        )
        sampels[i] = if accept_prob[i] <= acceptance_ratio
            new_state
        else
            curr_state
        end
        curr_state = sampels[i]
    end

    return sampels
end

"""
    MultivariateMHS(
        N::Int=1; 
        sampled_dist,
        dim::Int, 
        proposal_cov::Union{Nothing, Matrix_f64}, 
    )

 Многомерный Metropolis–Hastings сэмплер.

Arguments
≡≡≡≡≡≡≡≡≡

    - N: число сэмплируемых псевдослучайных чисел;

    - sampled_dist: функция сэмплируемого распределения;

    - dim: размерность сэмплируемого векторного пространства (dim >= 2); 

    - init_state: начальное значение сэмпла;

    - proposal_cov: ковариационная матрица функция плотности proposal.

Returns
≡≡≡≡≡≡≡

    -  Список сэмплируемых значений типа Float64.
"""
function MultivariateMHS(
    N::Int=1; 
    sampled_dist, 
    dim::Int, 
    init_state::Union{Nothing, Matrix_f64}=nothing, 
    proposal_cov::Union{Nothing, Matrix_f64}=nothing, 
)::Matrix_f64
    dim >= 2 || throw(ArgumentError("`dim` must be more or equal to 2."))
    isnothing(proposal_cov) && (proposal_cov = Matrix_f64(1.0I, dim, dim);)
    sampels = zeros(N, dim)
    curr_state = if isnothing(init_state)
        rand(1, dim)
    else
        init_state
    end
    accept_prob = rand(N)
    for i=1:N
        new_state = MultivariateGS(dim=dim, mu=curr_state, cov=proposal_cov)
        acceptance_ratio = min(
            1.0, 
            sampled_dist(new_state) / sampled_dist(curr_state),
        )
        sampels[i, 1:dim] = if accept_prob[i] <= acceptance_ratio
            new_state
        else
            curr_state
        end
        curr_state = sampels[i:i, 1:dim]
    end

    return sampels
end