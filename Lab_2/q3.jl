using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

c = [[59 41 59 26 53 58] 
     [93 93 23 84 62 27] 
     [79 38 32 79 52 2] 
     [97 41 97 16 69 39]
     [97 75 10 58 20 97]]

f = [0 86 28 3 48 25]
m = [0 34 21 17 56 79]

@variable(model, x[i=1:5, j=1:6], Bin)
@variable(model, y[i=1:6], Bin)

@objective(model, Min, sum(sum(x[i,j]*c[i, j] for j in 1:6) for i in 1:5 ) + sum(y[i]*f[i] for i in 1:6))

@constraint(model, con1[i = 1:5, j = 1:6], x[i, j] <= y[j])
@constraint(model, con2[i = 1:5], sum(x[i,j] for j in 1:6) == 1)
@constraint(model, sum(y[i]*m[i] for i in 1:6) <= 100)

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in 1:6
    println("y[$(i)] = ", value(y[i]))
end
println("-------")
for i in 1:5
    for j in 1:6
        println("x[$(i), $(j)] = ", value(x[i, j]))
    end
end