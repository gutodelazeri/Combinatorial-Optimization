using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

M = 5
N = 3 # be careful with small values of N. It is necessary that at least one '1' appears in each of the N columns
D = 3

#a = rand([0, 1], M, N)
#c = rand(1:1000, N)

a= [[1 0 0]
    [1 0 0]
    [0 1 0]
    [0 1 0]
    [0 0 1]]
c= [2 3 4]

@variable(model, x[i = 1:N], Bin)
@variable(model, 0 <= y)

@objective(model, Min, y)

@constraint(model, con1[i = 1:M], sum(a[i,j]*x[j] for j in 1:N) == 1)
@constraint(model, con2, sum(x[i] for i in 1:N) == D)
@constraint(model, con3[i = 1:N], y >= c[i]*x[i])


optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in 1:N
    println("x[$(i)] = ", value(x[i]))
end