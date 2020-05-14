using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

P = [1,2]
r = [40, 10]
l = [120, 80]
p = [20, 10]

@variable(model, 0 <= x[P])

@objective(model, Max, sum(x[i]*l[i] for i in P))

@constraint(model, con[i = P], x[i] <= r[i])
@constraint(model, sum(p[i]*x[i] for i in P) <= 500)

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in P
    println("x[$(i)] = ", value(x[i]))
end
