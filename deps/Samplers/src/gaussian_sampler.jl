
"""
    SimpleGS(N::Int=1; mu::Float64=0.0, sigma::Float64=1.0)

Одномерный гауссовский сэмплер.

Arguments
≡≡≡≡≡≡≡≡≡

    - N: число сэмплируемых псевдослучайных чисел;

    - mu: среднее сэмплируемого распредления;

    - sigma: среднеквадратичное отклонение сэмплируемого распредления.

Returns
≡≡≡≡≡≡≡

    -  Список сэмплируемых значений типа Float64.
"""
function SimpleGS(N::Int=1; mu::Float64=0.0, sigma::Float64=1.0)::Matrix_f64
    return mu .+ sigma*randn(N, 1)
end

"""
    MultivariateGS(
        N::Int=1; 
        mu::Matrix_f64=[0.0 0.0], 
        cov::Matrix_f64=[1.0 0.0; 0.0 1.0], 
    )

Многомерный гауссовский сэмплер.

Arguments
≡≡≡≡≡≡≡≡≡

    - N: число сэмплируемых псевдослучайных чисел;

    - dim: размерность сэмплируемого пространства;

    - mu: матрица средних значений компонентов;

    - cov: ковариационная матрица.

Returns
≡≡≡≡≡≡≡

    -  Список сэмплируемых значений типа Float64.
"""
function MultivariateGS(
    N::Int=1; 
    dim::Int=2, 
    mu::Matrix_f64=[0.0 0.0], 
    cov::Matrix_f64=[1.0 0.0; 0.0 1.0], 
)::Matrix_f64
    mu_size, cov_size = size(mu), size(cov)
    is_wrong_size = cov_size[1] != cov_size[2] || mu_size[1] != 1 || 
        mu_size[2] != dim
    is_wrong_size && throw(ArgumentError("Wrong matrix size: $(mu_size) and $(cov_size).")) 

    std_normal_sampels = randn(N, dim)
    std_matrix = cholesky(cov).U
    return mu .+ std_normal_sampels*std_matrix
end