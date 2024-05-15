
"""
    GaussianSampler(N::Int=1; mu::Float64=0.0, sigma::Float64=1.0)

Одномерный гауссовский сэмплер.

Arguments
≡≡≡≡≡≡≡≡≡

    - N: число сэмплируемых псевдослучайных чисел;

    - mu: среднее сэмплируемого распредления;

    - sigma: среднеквадратичное отклонение сэмплируемого распредления.

Returns
≡≡≡≡≡≡≡

    -  Список сэмплируемых значений типа Sampels.
"""
function GaussianSampler(N::Int=1; mu::Float64=0.0, sigma::Float64=1.0)::Sampels
    return mu .+ sigma*randn(N)
end