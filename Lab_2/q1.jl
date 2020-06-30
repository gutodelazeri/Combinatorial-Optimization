using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

m = [17, 8, 5, 8, 8]
c = [5, 5, 7, 7, 8]
l = [10, 11, 13, 17, 11]
d = [2, 8, 3, 1, 20]

@variable(model, x[i=1:5], Bin)
@variable(model, 0 <= u[i=1:5], Int)

@objective(model, Max, sum( (l[i]*u[i] - m[i]*(1 - x[i]) - c[i]*x[i]) for i in 1:5))

@constraint(model, con[i = 1:5], u[i] <= x[i]*d[i])
@constraint(model, sum(x[i] for i in 1:5) <= 3)
@constraint(model, sum(u[i] for i in 1:5) <= 10000)
@constraint(model, x[1]  == x[2])
@constraint(model, x[5]  <= x[4])

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in 1:5
    println("x[$(i)] = ", value(x[i]))
    println("u[$(i)] = ", value(u[i]))
    println("-----")
end