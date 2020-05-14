using JuMP
using GLPK
using GLPKMathProgInterface

model = Model()
set_optimizer(model, GLPK.Optimizer)

I = [1,2,3]
I′ = [1,2]
d = [[600,320,720], [440,260,560], [200,160,280]]
k = [[4,8,3],[8,13,10],[22,20,18]]

@variable(model, x[I,I])

@objective(model, Max, sum(sum(x[i,j]*d[i][j] for j in I) for i in I))

for j in I′
    @constraint(model, sum(x[i,3] + x[i,j] for i in I) <= 30)
end
for i in I
    for j in I
        @constraint(model, 0 <= x[i,j] <= k[i][j])
    end
end

optimize!(model)

println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
for i in I
    for j in I
        println("x[$(i),$(j)] = ", value(x[i,j]))
    end
end

