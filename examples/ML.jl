using PyPlot
import FractionalTimeDG
import SpecialFunctions

#T = Float64
T = BigFloat

MLneg_power_series = FractionalTimeDG.MLneg_power_series
MLneg_asymptotic_series = FractionalTimeDG.MLneg_asymptotic_series
MLneg_integral = FractionalTimeDG.MLneg_integral 

half = parse(T, "0.5")
two = parse(T, "2")
five = parse(T, "5")
ten = parse(T, "10")
twenty = parse(T, "20")
fifty = parse(T, "50")

α = half
λ = one(T)


tol = parse(T, "1e-14")

figure(1)
nmax = 20
x1 = range(zero(T), 1/two, length=201)
y1 = T[ MLneg_power_series(α, x, nmax, tol) for x in x1 ]
plot(x1, y1 - erfcx.(x1))
grid(true)

figure(2)
x2 = range(one(T), twenty, length=201)
t2 = (x2/λ).^(1/α) 
nmax = 15
y2 = T[ MLneg_integral(α, λ, t, nmax) for t in t2 ]
plot(x2, y2 - erfcx.(x2))
grid(true)

figure(3)
x3 = range(twenty, fifty, length=201)
nmax = 10
y3 = T[ MLneg_asymptotic_series(α, x, nmax) for x in x3 ]
plot(x3, y3 - erfcx.(x3))
grid(true)
