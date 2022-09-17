# Introduction
A boxed Jacobian constraint can be formulated as follows:

    u(t+1) = u(t) + M⁻¹·Jᵗ·λ + Δt·M⁻¹·f
    J·u(t+1) = RHS
    subject to λₗ ≤ λ ≤ λₕ 

The fixed pointer iterative solvers such as PGS (projected Gauss-Seidel) can  handle it directly by projecting (clamping) the intermediate λ within its allowed domain after each iteration.
However, it is not straightforward to handle it in the pivot-based solvers such as Lemke. In the following I show how to reformulate the boxed constraints in the same spirit to the dry-frictional constraints.
For that we need to add an extra condition on the domain λₕ ≤ λ ≤ λₗ.

    λₕ > 0 ⋏ λₗ < 0
In other words, λₕ must be strictly positive, and λₗ must be strictly negative.
WIth this restrictions the application of the constraints include the following:

* Contact boxed friction constraints
* Hinge/Slider friction constraints
* Hinge/Slider motors

# Reformulation of the constraints

First, we need to expand the equality constraint into two inequality constraints with corresponding lagrangian multipliers as follows:

     J·u(t+1) - RHS ≥ 0 with λ⁺ ≥ 0
    -J·u(t+1) + RHS ≥ 0 with λ⁻ ≥ 0

The corresponding lagrangian formation with the complementarity conditions will be:

    u(t+1) = u(t) + M⁻¹·Jᵗ·λ⁺ - M⁻¹·Jᵗ·λ⁻ + Δt·M⁻¹·f
           =  u(t) + M⁻¹·Jᵗ·(λ⁺ - λ⁻) + Δt·M⁻¹·f
     J·u(t+1) - RHS ≥ 0
    -J·u(t+1) + RHS ≥ 0
    subject to 0 ≤ λ⁺, 0 ≤ λ⁻
Therefore  λ = λ⁺ - λ⁻.

Next, we must limit the domain of λ = λ⁺ - λ⁻ such that  λₗ ≤ λ ≤ λₕ, i.e.:

    λ⁺ ≤ λₕ and λ⁻ ≤ (-λₗ)
At the same time, we need to allow one of the constraints to break if λ⁺ or λ⁻ is at its maximum.
For that we introduce two adidtional multipliers β⁺  and β⁻ as follows:

     J·u(t+1) - RHS + β⁺ ≥ 0 with λ⁺ ≥ 0
    -J·u(t+1) + RHS + β⁻ ≥ 0 with λ⁻ ≥ 0

At the same time we must to keep λ⁺ and λ⁻ in the domain:

    -λ⁺ + λₕ ≥ 0 with β⁺ ≥ 0
    -λ⁻ + (-λₗ) ≥ 0 with β⁻ ≥ 0
This can be loosely interpreted as follows.
If  β⁺  or β⁻ is activated, the applied force/torque induced by λ⁺ or λ⁻ must be at its limit λₕ or 
-λₗ.
More formally:

    J·u(t+1) - RHS + β⁺ = w₁⁺     - (1)
    0 ≤ λ⁺ ⊥ w₁⁺ ≥ 0              - (2)
    -λ⁺ + λₕ = w₂⁺                - (3)
    0 ≤ β⁺ ⊥ w₂⁺ ≥ 0              - (4)

    -J·u(t+1) + RHS + β⁻ = w₁⁻    - (5)
    0 ≤ λ⁻ ⊥ w₁⁻ ≥ 0              - (6)
    -λ⁻ + (-λₗ) = w₂⁻              - (7)
    0 ≤ β⁻ ⊥ w₂⁻ ≥ 0              - (8)

# Interpretatin of the reformulated constraints
Here we try to interpret the reformulation of (1), (2), (3), and(4) for each case of the complementarity. The same arguments apply to (5), (6), (7), and(8).

## Case 1⁺: λ⁺  = 0 (no induced force/torque applied)

    from (3), w₂⁺ = λₕ > 0.
    from (4), β⁺ = 0.
    from (1), J·u(t+1) - RHS = w₁⁺ ≥ 0
This implies the inqeuality holds without applying the force/torque induced by λ⁺.
This case is further handled on the negative side. See Case 2.1⁻ below.

## Case 2⁺: λ⁺  > 0

    from (2), w₁⁺ = 0.
    from (1), J·u(t+1) - RHS + β⁺ = 0.
We need to examine two subcases of Case2⁺.

### Case 2.1⁺.  λ⁺  > 0 ⋏ β⁺ > 0

    from (4), w₂⁺ = 0 .
    from (3), λ⁺ = λₕ > 0.
    from (2), w₁⁺ = 0.
    from (1),  J·u(t+1) - RHS + β⁺ = 0.
This implies `J·u(t+1) - RHS  < 0` and the original inequality does not hold.
For example, a solid object is sliding on a slope, even after the maximum induced force/torque with `λ⁺ = λₕ` is applied.

### Case 2.2⁺.  λ⁺  > 0 ⋏ β⁺ = 0
    from (2), w₁⁺ = 0.
    from (1), J·u(t+1) - RHS  = 0.
This implies the original equality holds.
    from (4),  w₂⁺ ≥ 0 .
    from (3), λ⁺ = λₕ - w₂⁺.
This implies `0 < λ⁺ ≤ λₕ`, and  λ⁺ is active and within its range.
For example, a solid object is holding on a slope with the friction force/torque applied.

## Case 1⁻: λ⁻  = 0 (no induced force/torque applied)

    from (7), w₂⁻ = -λₗ > 0.
    from (8), β⁻ = 0.
    from (5), -J·u(t+1) + RHS = w₁⁻ ≥ 0 thus J·u(t+1) - RHS = -w₁⁻ ≤ 0
This implies the inqeuality holds without applying the force/torque induced by λ⁺.
This case is further handled on the positive side. See Case 2.1⁺ above.

## Case 2⁻: λ⁻  > 0

    from (6), w₁⁻ = 0.
    from (5), -J·u(t+1) + RHS + β⁻ = 0.
We need to examine two subcases of Case2⁻.

### Case 2.1⁻.  λ⁻  > 0 ⋏ β⁻ > 0

    from (8), w₂⁻ = 0 .
    from (7), λ⁻ = -λₗ > 0.
    from (6), w₁⁻ = 0.
    from (5),  -J·u(t+1) + RHS + β⁺ = 0.
This implies `J·u(t+1) - RHS  > 0` and the original inequality does not hold.
For example, a solid object is sliding on a slope, even after the maximum induced force/torque with `λ⁻ = -λₗ` is applied.

### Case 2.2⁻.  λ⁻  > 0 ⋏ β⁻ = 0
    from (6), w₁⁻ = 0.
    from (5), -J·u(t+1) + RHS  = 0, thus J·u(t+1) - RHS  = 0.
This implies the original equality holds.
    from (4),  w₂⁻ ≥ 0 .
    from (3), λ⁻ = -λₕ - w₂⁻.
This implies `0 < λ⁻ ≤ -λₗ`, and  λ⁻ is active and within its range.
For example, a solid object is holding on a slope with the friction force/torque applied.


# LCP Formulation
The lagrangian with the complementarity conditions will be reformulated as follows:

    u(t+1) = u(t) + M⁻¹·Jᵗ·λ⁺ - M⁻¹·Jᵗ·λ⁻ + β⁺ + β⁻ + Δt·M⁻¹·f
    0 ≤ λ⁺ ⊥ ( J·u(t+1) - RHS + β⁺ ) ≥ 0
    0 ≤ λ⁻ ⊥ ( -J·u(t+1) + RHS + β⁻ ) ≥ 0
    0 ≤ β⁺ ⊥ (-λ⁺ + λₕ) ≥ 0
    0 ≤ β⁻ ⊥ (-λ⁻ + (-λₗ)) ≥ 0

Using the block matrix notations:

         | +J |      | -RHS |       |λ⁺ |       |β⁺|      | 1 0 |        | λₕ|
    J' = |    | q' = |      |  λ' = |   |  β' = |  |  I = |     |  Λ' =  |   |
         | -J |      | +RHS |       |λ⁻ |       |β⁻|      | 0 1 |        |-λₗ|
then

    u(t+1) = u(t) + M⁻¹·J'ᵗ·λ' + I·β' + Δt·M⁻¹·f - (9)
    J'·u(t+1) ≥ -q' - I·β' -(10)
    -λ' + Λ' ≥ 0 - (11)

from (9)

    J'·u(t+1) = J'·u(t) + J'·M⁻¹·J'ᵗ·λ' + Δt·J'·M⁻¹·f - (9)
from (10)

    J'·u(t) + J'·M⁻¹·J'ᵗ·λ' + Δt·J'·M⁻¹·f  ≥ -q' - I·β' 
and

    [J'·M⁻¹·J'ᵗ·λ' + I·β'] + [J'·u(t)  + Δt·J'·M⁻¹·f  +q']  ≥ 0 - (12)
and from (11) and (12),

    [J'·M⁻¹·J'ᵗ·λ' + I·β'] + [J'·u(t)  + Δt·J'·M⁻¹·f  +q']  ≥ 0
    [-I·λ'] + [Λ'] ≥ 0 
combined,

    | J'·M⁻¹·J'ᵗ |  +I | | λ'|   |J'·u(t)  + Δt·J'·M⁻¹·f  +q'|  | 0 |
    | ---------- + --- |·| - | + | ------------------------- | ≥|   |
    |  -I        |   0 | | β'|   |  Λ'                       |  | 0 |
    
This is the final LCP forumulation to be fed to the solvers.
    
    
    