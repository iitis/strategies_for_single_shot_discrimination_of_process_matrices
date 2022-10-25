using Convex, SCS
using LinearAlgebra
using QuantumInformation

const MOI = Convex.MOI

swap = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
permutation23 = I(2) ⊗ swap ⊗ I(2)
permutation12 = swap ⊗ I(4)
permutation34 =  I(4) ⊗ swap

function change_system(W)

    """ Function changing the systems AI -> BI and AO -> BO. """

    d = Int(size(W)[1]^(1/4))

    return permutesystems(W, fill(d, 4), [3,4,1,2])
end


function change_probability_to_distance(x)

    """ Function changing the probability of correct discrimination to distance. """

    return 4 * x - 2
end


function sdp_distance_for_free_process_matrices(W0)

    """SDP calculating the distance between a process matrix W0 and the set W^(A || B). """

    d = Int(size(W0)[1]^(1/4))
    Y0 = ComplexVariable(d^3,d^3)
    Y1 = ComplexVariable(d^3, d^3)
    rho = ComplexVariable(d^2, d^2)
    alpha = Variable()


    constraints = [Y0 == Y0']
    constraints += [Y1 == Y1']

    constraints += [rho in :SDP]
    constraints += [tr(rho) == 1]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W0 in :SDP]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * permutation23 * (rho ⊗ I(d^2) ) * permutation23' in :SDP]



    f = real(alpha * d^2)
    problem = minimize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )

    return change_probability_to_distance(problem.optval)
end


function sdp_distance_for_comb_process_matrices(W0)

    """SDP calculating the distance between a process matrix W0 and the set W^(A < B). """

    d = Int(size(W0)[1]^(1/4))
    Y0 = ComplexVariable(d^3,d^3)
    Y1 = ComplexVariable(d^3, d^3)
    W = ComplexVariable(d^3, d^3)
    alpha = Variable()


    constraints = [Y0 == Y0']
    constraints += [Y1 == Y1']

    constraints += [W in :SDP]
    constraints += [tr(W) == d]

    constraints += [partialtrace(W,3,fill(d,3)) == partialtrace(partialtrace(W,3,fill(d,3)),2,fill(d,2))  ⊗ I(d)/d]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W0 in :SDP]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W ⊗ I(d) in :SDP]



    f = real(alpha * d^2)
    problem = minimize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )

    return change_probability_to_distance(problem.optval)
end


function sdp_distance_for_sep_process_matrices(W0)

    """SDP calculating the distance between a process matrix W0 and the set W^SEP. """

    d = Int(size(W0)[1]^(1/4))
    Y0 = ComplexVariable(d^3,d^3)
    Y1 = ComplexVariable(d^3, d^3)
    WAB = ComplexVariable(d^3, d^3)
    WBA = ComplexVariable(d^3, d^3)
    alpha = Variable()
    p = Variable()

    constraints = [p >= 0]
    constraints += [Y0 == Y0']
    constraints += [Y1 == Y1']

    constraints += [WAB in :SDP]
    constraints += [WBA in :SDP]

    constraints += [partialtrace(WAB,3,fill(d,3)) == partialtrace(partialtrace(WAB,3,fill(d,3)),2,fill(d,2))  ⊗ I(d)/d]
    constraints += [partialtrace(WBA,3,fill(d,3)) == partialtrace(partialtrace(WBA,3,fill(d,3)),2,fill(d,2))  ⊗ I(d)/d]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 * W0 in :SDP]

    constraints += [permutation12 * (I(d) ⊗ Y0) * permutation12' - I(d^2)/d ⊗ partialtrace(Y0,1,fill(d,3)) + Y1 ⊗ I(d) - partialtrace(Y1, 3,fill(d,3)) ⊗ I(d^2)/d + alpha * I(d^4) - 1/2 *(WAB ⊗ I(d) +  permutation23* permutation34 * permutation12 * permutation23 * (WBA ⊗ I(d)) * permutation23' * permutation12' * permutation34' * permutation23') in :SDP]

    constraints += [tr(WAB + WBA) == d]


    f = real(alpha * d^2)
    problem = minimize(f, constraints)
    solve!(
        problem,
        MOI.OptimizerWithAttributes(SCS.Optimizer, "eps_abs" => 1e-8, "eps_rel" => 1e-8);
        silent_solver = true
    )
    return change_probability_to_distance(problem.optval)
end


WCNS = 1/4 * (I(16) + 1/sqrt(2) * (I(2) ⊗ sz ⊗ sz ⊗ I(2) + sz ⊗ I(2) ⊗ sx ⊗ sz))

@show(sdp_distance_for_free_process_matrices(WCNS)) # The distance between W^CNS and the set of all free process matrices

@show(sdp_distance_for_comb_process_matrices(WCNS)) # The distance between W^CNS and the set of combs A < B

@show(sdp_distance_for_comb_process_matrices(change_system(WCNS))) # The distance between W^CNS and the set of combs B < A

@show(sdp_distance_for_sep_process_matrices(WCNS)) # The distance between W^CNS and the set of all separable process matrices

