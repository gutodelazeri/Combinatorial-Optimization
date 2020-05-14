using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

P = [1,2]
r = [6000.0, 4000.0]
l = [25, 30]
p = [200,140]

@variable(model, 0 <= x[P])

@objective(model, Max, sum(x[i]*l[i] for i in P))

@constraint(model, con[i = P], x[i] <= r[i])
@constraint(model, sum((1/p[i])*x[i] for i in P) <= 40)

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in P
    println("x[$(i)] = ", value(x[i]))
end
