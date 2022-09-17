# Introduction
A bilateral (free) Jacobian constraint can be formulated as follows:

    u(t+1) = u(t) + M⁻¹·Jᵗ·λ + Δt·M⁻¹·f
    J·u(t+1) = RHS
    λ : free

The fixed pointer iterative solvers such as PGS (projected Gauss-Seidel) can  handle it directly.
However, the pivot-based solvers such as Lemke can not take them directly. 
There are two ways to handle this. One is to convert the equality to
two sets of inqualities. The other is to reformulate the problem bi removing the multipliers for the bi-lateral constraints by substitution. It involves generation of a matrix inverse and we use Cholesky factorization for it.

## Type 1: Conversion two Inequalities.

The bilateral constraint can be represented by two inequalities as follows.

    u(t+1) = u(t) + M⁻¹·Jᵗ·λ⁺ - M⁻¹·Jᵗ·λ⁻ + Δt·M⁻¹·f
           = u(t) + M⁻¹·Jᵗ·(λ⁺ - λ⁻) + Δt·M⁻¹·f
           = u(t) + M⁻¹·Jᵗ·λ + Δt·M⁻¹·f
        where λ = λ⁺ - λ⁻.
    s.t.
        0 ≤ ( J·u(t+1) - RHS ) ⊥ λ⁺ ≥ 0
        0 ≤ (-J·u(t+1) + RHS ) ⊥ λ⁻ ≥ 0

Practically we should add an epsilon for numerical stability as follows.

        0 ≤ ( J·u(t+1) - RHS + ε ) ⊥ λ⁺ ≥ 0
        0 ≤ (-J·u(t+1) + RHS - ε ) ⊥ λ⁻ ≥ 0

## Type 2: Reformulation of the problem.

We consider the following MLCP problem (after Schur's complement is applied to eliminate the u(t) term from the problems).

        | A | B | |Zb | |qb | = | 0 | - bilateral part
        |---+---|·| - |+| - |   | - |
        | C | D | |Zu | |qz | ≥ | Wu| - unilateral part
 
This can be decomposed into the following two sets of the equations.

    A·Zb + B·Zu + qb = 0  - (1)
    C·Zb + D·Zu + qz ≥ Wu - (2)

from (1)

    Zb = -A⁻¹·B·Zu - A⁻¹·qb
substitute Zb in (2) we get:

    -C·A⁻¹·B·Zu - C·A⁻¹·qb + D·Zu + qz ≥ Wu
and    
    
    (-C·A⁻¹·B + D)·Zu + [-C·A⁻¹·qb+ qz] = Wu
    s.t.  0 ≤ Zu ⊥ Wu ≥ 0  
This is the LCP problem to be solved by the pivot-based solvers.
This requires the inverse of `A`, which is most efficiently solved by LAPACK/BLAS's Cholesky solver.
