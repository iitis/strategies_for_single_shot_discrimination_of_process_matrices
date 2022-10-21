using Convex, SCS
using LinearAlgebra
using QuantumInformation

const MOI = Convex.MOI

# W \in AI ⊗ A0 ⊗ BI ⊗ B0, where we assume that dimension of all spaces A0, AI, B0, BI equals d

function sdp(W0, W1)

    """SDP calculating the probability of correct discrimination between W0 and W1 - primal problem. """

    d = Int(size(W0)[1]^(1/4))

    M0 = ComplexVariable(size(W0))
    M1 = ComplexVariable(size(W1))

    NS = M0 + M1

    constraints = [M0 in :SDP]

    constraints += [M1 in :SDP]

    constraints += [partialtrace(NS, 2, fill(d, 4)) == I(d)/d ⊗ partialtrace(partialtrace(NS, 2, fill(d, 4)), 1, fill(d, 3))]

    constraints += [partialtrace(NS, 4, fill(d, 4)) == partialtrace(partialtrace(NS, 4, fill(d, 4)), 3, fill(d, 3)) ⊗ I(d)/d]

    constraints += [tr(NS) == d^2]


    f = real(1/2 * (tr(W0 * M0) + tr(W1 * M1)))
    problem = maximize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-5);
        silent_solver = true
    )
    return problem.optval
end
