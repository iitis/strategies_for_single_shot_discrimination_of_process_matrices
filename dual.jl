using Convex, SCS
using LinearAlgebra
using QuantumInformation

include("sdpcomb.jl")
include("sdp.jl")

swap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
permutation23 = I(2) ⊗ swap ⊗ I(2)
permutation12 = swap ⊗ I(4)
permutation34 =  I(4) ⊗ swap


const MOI = Convex.MOI

# W \in AI ⊗ A0 ⊗ BI ⊗ B0, where we assume that dimension of all spaces A0, AI, B0, BI equals d

function sdp_dual(W0, W1)

    """ SDP calculating the probability of correct discrimination between W0 and W1 - dual problem. """

    d = Int(size(W0)[1]^(1/4))

    Y0 = ComplexVariable(d^3,d^3)
    Y1 = ComplexVariable(d^3, d^3)
    alpha = Variable()


    constraints = [Y0 == Y0']

    constraints += [Y1 == Y1']

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W0 in :SDP]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W1 in :SDP]



    f = real(alpha * d^2)
    problem = minimize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-5);
        silent_solver = true
    )

    return problem.optval
end
