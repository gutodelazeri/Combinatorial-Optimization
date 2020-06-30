using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer);

@variable(model, x[i=1:8, j=1:8], Bin)

@objective(model, Min, sum(sum(x[i,j] for j in 1:8) for i in 1:8 ))

@constraint(model, con1[i = 1:8], sum(x[i,j] for j in 1:8) == 4)
@constraint(model, con2[j = 1:8], sum(x[i,j] for i in 1:8) == 4)
@constraint(model, con3[i = 1:8, j = 1:5], 1 <= x[i, j] + x[i, j + 1] + x[i, j + 2] <= 2)
@constraint(model, con4[i = 1:5, j = 1:8], 1 <= x[i, j] + x[i+1, j] + x[i+2, j] <= 2)

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in 1:8
    println("$(value(x[i, 1]))  $(value(x[i, 2]))  $(value(x[i, 3]))  $(value(x[i, 4]))  $(value(x[i, 5]))  $(value(x[i, 6]))  $(value(x[i, 7]))  $(value(x[i, 8]))")
end