# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import sympy
import re
import traceback

# --- 1. A Simple and Direct Parser ---
def normalize_expression(expr: str) -> str:
    """A simple parser that directly prepares the string for SymPy.

    Keeps normalization conservative: maps common unicode characters,
    removes surrounding whitespace but preserves token boundaries so
    later regex-based substitutions work reliably.
    """
    expr = expr.strip()
    # Map a few common unicode characters to ASCII equivalents
    char_map = {'²': '^2', '³': '^3', '’': "'", '√': 'sqrt'}
    for k, v in char_map.items():
        expr = expr.replace(k, v)

    # Don't aggressively remove spaces here; use targeted regex replacements later.
    return expr

# --- 2. The First-Order Linear Solver ---
def _solve_linear_step_by_step(eq, y, x):
    """Generates steps for first-order linear equations."""
    dydx = y.diff(x)
    rearranged = (eq.lhs - eq.rhs).expand()
    coeff_dydx = rearranged.coeff(dydx, 1)
    if coeff_dydx != 0:
        rearranged = rearranged / coeff_dydx
    p_x = rearranged.coeff(y, 1)
    q_x = -(rearranged - dydx - p_x * y)
    steps = [{
        "rule_name": "Identify P(x) and Q(x)",
        "result": f"P(x) = {sympy.latex(p_x)}, Q(x) = {sympy.latex(q_x)}",
        "explanation": "The equation is a first-order linear ODE: y' + P(x)y = Q(x)."
    }]
    integral_p = sympy.integrate(p_x, x)
    integrating_factor = sympy.exp(integral_p)
    steps.append({
        "rule_name": "Calculate Integrating Factor (I.F.)",
        "result": f"I(x) = e^{{\int P(x) dx}} = {sympy.latex(integrating_factor)}",
        "explanation": "The integrating factor is e raised to the integral of P(x)."
    })
    integral_of_rhs = sympy.integrate(integrating_factor * q_x, x)
    C = sympy.Symbol('C')
    final_solution = sympy.Eq(y, (integral_of_rhs + C) / integrating_factor)
    steps.append({
        "rule_name": "Integrate and Solve for y",
        "result": sympy.latex(sympy.Eq(integrating_factor * y, integral_of_rhs + C)),
        "explanation": "The solution is found by the formula y * I(x) = ∫ Q(x) * I(x) dx."
    })
    steps.append({
        "rule_name": "Final Solution",
        "result": sympy.latex(final_solution),
        "explanation": "Isolating y gives the general solution."
    })
    return {"solution_summary": sympy.latex(final_solution), "steps": steps}

# --- 3. A Corrected Classifier ---
def _is_linear(eq, y, x):
    """Helper to check for first-order linearity y' + P(x) y = Q(x)."""
    try:
        dydx = y.diff(x)
        # Reject if higher derivatives are present
        if not eq.has(dydx) or any(eq.has(y.diff(x, i)) for i in range(2, 5)):
            return False

        rearranged = eq.lhs - eq.rhs
        collected = sympy.collect(rearranged, [y, dydx])
        coeff_dydx = collected.coeff(dydx, 1)
        if coeff_dydx == 0:
            return False

        normalized_collected = (collected / coeff_dydx).expand()
        p_x = normalized_collected.coeff(y, 1)
        q_x_negative = normalized_collected.subs([(y, 0), (dydx, 0)])

        return not p_x.has(y) and not q_x_negative.has(y)
    except Exception:
        return False

# --- Add: Constant-coefficient linear ODE detector & stepwise solver ---
def _is_constant_coeff_linear(eq, y, x, max_order_check=8):
    """Return order if eq is linear with constant coefficients, else 0."""
    try:
        rearr = (eq.lhs - eq.rhs).expand()
        # find maximum derivative order present (up to max_order_check)
        max_order = 0
        for i in range(1, max_order_check + 1):
            if rearr.has(y.diff(x, i)):
                max_order = max(max_order, i)
        # require at least second order for 'higher-order' detection
        if max_order < 2:
            return 0
        # check coefficients are constants (no x)
        for k in range(0, max_order + 1):
            coeff = rearr.coeff(y.diff(x, k))
            # if coefficient depends on x or y, not constant-coeff linear
            if coeff != 0 and coeff.has(x):
                return 0
        return max_order
    except Exception:
        return 0

def _build_char_poly_and_coeffs(eq, y, x, order):
    """Return tuple (coeffs_list, r_poly) where coeffs_list is [a_order,...,a0]."""
    rearr = (eq.lhs - eq.rhs).expand()
    coeffs = []
    r = sympy.Symbol('r')
    for k in range(order, -1, -1):
        coeff = sympy.simplify(rearr.coeff(y.diff(x, k)))
        coeffs.append(sympy.simplify(coeff))
    # poly: a_n*r^n + ... + a0
    poly = sum(coeffs[i] * r**(order - i) for i in range(len(coeffs)))
    poly = sympy.simplify(poly)
    return coeffs, sympy.expand(poly), r

def _compose_cf_from_roots(roots_dict, r):
    """
    Build complementary function expression and a list of LaTeX step strings.

    Inputs:
      - roots_dict: dict returned by sympy.roots(poly, r) => {root: multiplicity}
      - r: sympy Symbol used for characteristic polynomial (kept for compatibility)

    Returns:
      (CF_expr, cf_steps) where CF_expr is a sympy expression and cf_steps is
      a list of LaTeX strings describing each root's contribution.
    """
    x = sympy.Symbol('x')
    terms = []
    cf_steps = []
    C_counter = 1
    handled = set()

    for root, mult in roots_dict.items():
        # avoid processing conjugates twice
        if root in handled:
            continue

        # Simplify root first
        root_s = sympy.simplify(root)

        # real root (imaginary part zero)
        if sympy.im(root_s) == 0:
            root_val = sympy.simplify(sympy.re(root_s))
            for k in range(mult):
                Ci = sympy.Symbol(f'C{C_counter}')
                term = Ci * x**k * sympy.exp(root_val * x)
                terms.append(term)

                # LaTeX: explicit spacing and \text{} to avoid word-joining
                step_latex = (
                    r"\text{Real root } " + sympy.latex(root_val)
                    + r"\text{ with multiplicity } " + str(mult)
                    + r"\text{: contributes } " + sympy.latex(term)
                )
                cf_steps.append(step_latex)
                C_counter += 1

        else:
            # complex root a + i b -> produce cos/sin pair; show b as positive magnitude
            a = sympy.simplify(sympy.re(root_s))
            b = sympy.simplify(sympy.im(root_s))
            b_abs = sympy.Abs(b)  # ensures we print positive magnitude inside ±

            pair_terms = []
            for k in range(mult):
                A = sympy.Symbol(f'C{C_counter}'); C_counter += 1
                B = sympy.Symbol(f'C{C_counter}'); C_counter += 1
                term = x**k * sympy.exp(a * x) * (A * sympy.cos(b_abs * x) + B * sympy.sin(b_abs * x))
                # Note: we use b_abs in the trig argument so it shows as \sqrt{2} (positive)
                pair_terms.append(term)
                terms.append(term)

            # LaTeX: clear wording, positive imaginary magnitude inside ±
            step_latex = (
                r"\text{Complex conjugate roots } "
                + sympy.latex(a) + r" \pm " + sympy.latex(b_abs) + r"i"
                + r"\text{ with multiplicity } " + str(mult)
                + r"\text{: contributes } "
                + " + ".join(sympy.latex(t) for t in pair_terms)
            )
            cf_steps.append(step_latex)

            # mark conjugate as handled too
            handled.add(sympy.conjugate(root_s))

        handled.add(root_s)

    CF = sympy.Add(*terms) if terms else sympy.Integer(0)
    return CF, cf_steps

def _guess_trial_for_single_term(rhs_term, order, roots_dict):
    """Generate trial for single additive RHS term. Returns (trial_expr, unknown_symbols)."""
    x = sympy.Symbol('x')
    # polynomial RHS: degree d -> trial polynomial degree d with undetermined coeffs
    if rhs_term.is_polynomial(x):
        poly = sympy.Poly(rhs_term, x)
        d = poly.degree()
        coeffs = sympy.symbols('a0:%d' % (d + 1))
        trial = sum(coeffs[i] * x**i for i in range(d + 1))
        # shift for root r=0
        if 0 in roots_dict:
            shift = roots_dict[0]
            trial = x**shift * trial
        return trial, list(coeffs)

    # exponential RHS e^{a x} * poly(x)
    if rhs_term.has(sympy.exp) and rhs_term.is_Mul:
        exp_factors = [f for f in rhs_term.args if f.has(sympy.exp)]
        other = sympy.Mul(*[f for f in rhs_term.args if not f.has(sympy.exp)])
        exp_part = exp_factors[0]
        if exp_part.func == sympy.exp:
            a_expr = exp_part.args[0]
            # try to represent a as scalar (not depending on x)
            if a_expr.has(sympy.Symbol('x')):
                # more complex exp, give up
                return None, None
            a = sympy.simplify(a_expr)
            deg = 0
            if other != 1 and other.is_polynomial(sympy.Symbol('x')):
                deg = sympy.Poly(other, x).degree()
            coeffs = sympy.symbols('b0:%d' % (deg + 1))
            poly_part = sum(coeffs[i] * x**i for i in range(deg + 1))
            s = 0
            for root, m in roots_dict.items():
                if sympy.simplify(root - a) == 0:
                    s = m
            trial = x**s * sympy.exp(a * x) * poly_part
            return trial, list(coeffs)

    # sinusoidal RHS: A*cos(bx)+B*sin(bx) or just cos/sin
    if rhs_term.has(sympy.sin) or rhs_term.has(sympy.cos):
        # attempt to find frequency(s); we only handle simple sin(bx) or cos(bx)
        # find first trig function
        trig = None
        add_args = rhs_term.args if rhs_term.is_Add else [rhs_term]
        for t in add_args:
            if t.has(sympy.sin) or t.has(sympy.cos):
                trig = t
                break
        if trig is None:
            trig = rhs_term
        # get the argument of sin/cos if possible
        funcs = [f for f in trig.atoms(sympy.Function) if isinstance(f, sympy.Function)]
        # fallback frequency
        b = sympy.Integer(1)
        # try to find numeric frequency inside sin(arg) or cos(arg)
        for f in trig.atoms(sympy.Function):
            if f.func == sympy.sin or f.func == sympy.cos:
                arg = f.args[0]
                # arg typically like b*x
                if arg.has(sympy.Symbol('x')):
                    b_candidate = sympy.simplify(arg / sympy.Symbol('x'))
                    if not b_candidate.has(sympy.Symbol('x')):
                        b = b_candidate
                break
        A, B = sympy.symbols('A B')
        s = 0
        # check if imaginary roots with magnitude b exist -> resonance shift
        for root, m in roots_dict.items():
            if sympy.simplify(sympy.im(root)) != 0:
                if sympy.simplify(abs(sympy.im(root)) - abs(b)) == 0:
                    s = m
        trial = x**s * (A * sympy.cos(b * sympy.Symbol('x')) + B * sympy.sin(b * sympy.Symbol('x')))
        return trial, [A, B]

    return None, None

def _guess_trial_for_rhs(rhs, order, roots_dict):
    """
    Try to build a combined trial for additive RHS: if RHS is a sum, build trial per term.
    Returns (trial_expr_combined, unknowns_combined) or (None, None) if not recognized.
    """
    # If RHS is an Add, handle termwise
    if rhs.is_Add:
        trials = []
        unknowns = []
        for term in rhs.args:
            tr, u = _guess_trial_for_single_term(term, order, roots_dict)
            if tr is None:
                # if any term is unrecognized, abandon combined approach
                return None, None
            trials.append(tr)
            unknowns.extend(u)
        return sympy.simplify(sympy.Add(*trials)), unknowns

    # single-term RHS
    return _guess_trial_for_single_term(rhs, order, roots_dict)

def _solve_constant_coeff_linear(eq, y, x, order):
    """Solve constant-coefficient linear ODE and produce step-by-step solution."""
    steps = []
    # Build characteristic polynomial
    coeffs, poly, r = _build_char_poly_and_coeffs(eq, y, x, order)
    steps.append({"rule_name": "Form Characteristic Polynomial",
                  "result": sympy.latex(poly),
                  "explanation": "Form the polynomial a_n r^n + ... + a_0 = 0 from coefficients."})

    # find roots with multiplicity
    roots_dict = sympy.roots(poly, r)  # returns dict {root:mult}
    factored = sympy.factor(poly)
    steps.append({"rule_name": "Factor / Roots",
                  "result": sympy.latex(factored),
                  "explanation": f"Roots (with multiplicity): {', '.join(f'{sympy.latex(k)}: {v}' for k,v in roots_dict.items())}"})

    # Complementary Function
    CF, cf_steps = _compose_cf_from_roots(roots_dict, r)
    for s in cf_steps:
        steps.append({"rule_name": "CF Step", "result": s, "explanation": ""})
    steps.append({"rule_name": "Complementary Function", "result": sympy.latex(sympy.Eq(y, CF)), "explanation": "Combine root contributions."})

    # Now attempt particular integral if non-homogeneous
    rhs = sympy.simplify(eq.rhs)
    if rhs == 0:
        final = sympy.Eq(y, CF)
        steps.append({"rule_name": "Final Solution (Homogeneous)", "result": sympy.latex(final), "explanation": ""})
        return {"solution_summary": sympy.latex(final), "steps": steps}

    # try to build a trial for undetermined coefficients (supports additive RHS)
    trial, unknowns = _guess_trial_for_rhs(rhs, order, roots_dict)
    if trial is None:
        # fallback: use dsolve to get particular solution (no stepwise PI)
        sol = sympy.dsolve(eq, y)
        steps.append({"rule_name": "Particular Integral (fallback)", "result": sympy.latex(sol.rhs - CF),
                      "explanation": "Automatic solver produced a particular integral (no undetermined-coeff steps)."})
        steps.append({"rule_name": "General Solution", "result": sympy.latex(sol), "explanation": ""})
        return {"solution_summary": sympy.latex(sol), "steps": steps}

    # ensure unknowns list
    if unknowns is None or len(unknowns) == 0:
        unknowns = list(sympy.symbols('p0:3'))

    # substitute trial into LHS operator: compute L[trial]
    rearr = (eq.lhs - eq.rhs).expand()
    n = order
    L_trial = sum(coeffs[i] * sympy.diff(trial, x, n - i) for i in range(len(coeffs)))
    L_trial = sympy.simplify(L_trial)

    # Form equation L_trial = RHS and attempt to solve for unknowns.
    eq_to_solve = sympy.simplify(sympy.expand(L_trial - rhs))

    soln = None
    try:
        # Case 1: polynomial in x -> coefficient matching
        if eq_to_solve.is_polynomial(x):
            poly_eq = sympy.Poly(eq_to_solve, x)
            deg = poly_eq.degree()
            equations = [sympy.Eq(poly_eq.coeff_monomial(x**k), 0) for k in range(0, deg + 1)]
            soln = sympy.solve(equations, unknowns, dict=True)
        else:
            # Case 2: try linear system extraction using linear_eq_to_matrix on a set of basis functions.
            # We'll expand eq_to_solve and collect terms grouped by independent functions (monomials, sin/cos, exp).
            expanded = sympy.expand(L_trial - rhs)
            # Get list of candidate basis functions from expanded expression (terms)
            terms = sympy.Add.make_args(expanded)
            # Create symbolic coefficients by treating unknowns as variables and matching the numeric coefficients
            # Build equations by projecting onto a set of sample functions produced by replacing unknowns by symbols.
            # Practical approach: generate equations by equating expanded expression evaluated at multiple x values
            # (numeric sampling) to produce linear equations for unknowns when symbolic matching fails.
            # Try symbolic linear solve first:
            soln = sympy.solve(sympy.Eq(L_trial, rhs), unknowns, dict=True)
            # If that didn't return usable result, try numeric sampling fallback to produce linear system.
            if not soln:
                # numeric sampling: pick distinct numeric x values avoiding singularities
                sample_points = [1, 2, 3, 5, 7][:max(1, min(5, len(unknowns)+1))]
                A_rows = []
                b_vec = []
                # prepare lambda for numeric evaluation
                for xv in sample_points:
                    subs_map = {x: sympy.Integer(xv)}
                    # compute numeric value of each unknown's contribution: differentiate trial wrt each unknown
                    row = []
                    for u in unknowns:
                        # partial derivative of L_trial wrt unknown u then numeric evaluation
                        part = sympy.simplify(sympy.diff(L_trial, u))
                        row.append(float(part.subs(subs_map).evalf()))
                    # compute RHS numeric
                    rhs_val = float(sympy.simplify(rhs).subs(subs_map).evalf())
                    # compute constant term (L_trial with unknowns = 0)
                    const_term = float(L_trial.subs({uu: 0 for uu in unknowns}).subs(subs_map).evalf())
                    A_rows.append(row)
                    b_vec.append(rhs_val - const_term)
                # solve linear system numerically (least squares)
                import numpy as _np
                A = _np.array(A_rows, dtype=float)
                b = _np.array(b_vec, dtype=float)
                try:
                    sol_numeric, *_ = _np.linalg.lstsq(A, b, rcond=None)
                    soln_dict = {unknowns[i]: sympy.N(sol_numeric[i]) for i in range(len(unknowns))}
                    soln = [soln_dict]
                except Exception:
                    soln = None
    except Exception:
        soln = None

    if not soln:
        # fallback to dsolve if solving for coefficients fails
        try:
            sol = sympy.dsolve(eq, y)
        except Exception as e:
            steps.append({"rule_name": "Particular Integral (fallback failure)", "result": str(e),
                          "explanation": "Both undetermined-coeff approach and automatic solver failed."})
            return {"error": "Failed to find particular integral."}
        steps.append({"rule_name": "Particular Integral (fallback)", "result": sympy.latex(sol.rhs - CF),
                      "explanation": "Attempt to solve undetermined coefficients failed; used automatic solver."})
        steps.append({"rule_name": "General Solution", "result": sympy.latex(sol), "explanation": ""})
        return {"solution_summary": sympy.latex(sol), "steps": steps}

    # take first solution dict (may be numeric approximations)
    soln = soln[0]
    # build particular solution
    y_part = sympy.simplify(trial.subs(soln))
    steps.append({"rule_name": "Trial Particular Solution", "result": sympy.latex(trial),
                  "explanation": "Assume trial solution with undetermined coefficients."})
    steps.append({"rule_name": "Solve for coefficients", "result": sympy.latex(soln),
                  "explanation": "Equate L(trial) to RHS and solve linear equations for unknown coefficients."})
    steps.append({"rule_name": "Particular Integral", "result": sympy.latex(y_part),
                  "explanation": "Substitute solved coefficients into the trial to obtain a particular integral."})

    general = sympy.simplify(CF + y_part)
    final = sympy.Eq(y, general)
    steps.append({"rule_name": "General Solution", "result": sympy.latex(final), "explanation": ""})
    return {"solution_summary": sympy.latex(final), "steps": steps}

# --- First-order higher-degree detector & solver (Clairaut + solvable-for-p) --- (unchanged) ---
def _is_first_order_higher_degree(eq, y, x):
    """
    Detect first-order 'higher-degree' equations.

    Returns:
      - None if not detected,
      - {'type': 'clairaut', 'f': f_p} for Clairaut form y = x*p + f(p) (p = y'),
      - {'type': 'solvable_for_p', 'p_solutions': [sol1, sol2, ...], 'deg_p': deg} if the
         algebraic equation F(x,y,p)=0 is non-linear in p (degree >= 2) and we can
         obtain algebraic branches for p.

    The detector is conservative: it will *not* return solvable_for_p for
    linear-in-p cases (degree == 1). Linear-in-p ODEs should be handled by the
    first-order linear classifier.
    """
    try:
        # Disallow higher derivatives (we only want first-order)
        if any((eq.lhs - eq.rhs).has(y.diff(x, i)) for i in range(2, 6)):
            return None

        p = sympy.Symbol('p')
        # Replace derivative by p to get algebraic F(x,y,p)
        F = sympy.simplify((eq.lhs - eq.rhs).subs(sympy.Derivative(y, x), p))

        # Check Clairaut: F == y - x*p - f(p)  <=>  F - y + x*p depends only on p
        G = sympy.simplify(F - y + x * p)
        free_non_p = set(sym for sym in G.free_symbols if sym != p)
        if len(free_non_p) == 0:
            f_p = sympy.simplify(G)
            return {'type': 'clairaut', 'f': f_p}

        # If F doesn't actually contain p, nothing to do
        if not F.has(p):
            return None

        # Try to determine degree in p. Prefer Poly; fallback to diff test.
        deg_p = None
        try:
            poly_p = sympy.Poly(sympy.expand(F), p)
            deg_p = int(poly_p.degree())
        except Exception:
            # fallback: compute second derivative wrt p; if it's zero -> linear or constant
            try:
                second = sympy.simplify(sympy.diff(F, p, 2))
                if second == 0:
                    deg_p = 1
                else:
                    deg_p = 2  # treat as non-linear (degree >=2)
            except Exception:
                deg_p = None

        # Only treat as solvable-for-p if truly nonlinear in p (degree >= 2).
        # Linear-in-p should be handled by _is_linear instead.
        if deg_p is not None and deg_p <= 1:
            return None

        # If degree unknown, still attempt solve but be conservative:
        try:
            sols = sympy.solve(sympy.Eq(F, 0), p, dict=False)
            sols = [sympy.simplify(s) for s in sols if s is not None]
            # If solutions exist and at least one is nonlinear-ish (contains p originally),
            # return them — but ensure it's not the trivial linear case.
            if sols:
                # If every sol is simply an expression independent of p and looks linear-in-p,
                # we still want to avoid capturing trivial linear cases. We make a best-effort:
                all_linear_forms = True
                for s in sols:
                    if s.has(p):
                        all_linear_forms = False
                        break
                if deg_p is None:
                    # couldn't determine degree by poly, but we have algebraic sols; treat as solvable_for_p
                    return {'type': 'solvable_for_p', 'p_solutions': sols, 'deg_p': deg_p}
                else:
                    if deg_p >= 2:
                        return {'type': 'solvable_for_p', 'p_solutions': sols, 'deg_p': deg_p}
        except Exception:
            return None

        return None
    except Exception:
        return None

def _solve_first_order_higher_degree(eq, y, x, parsed):
    """
    Solve first-order higher-degree equations detected by _is_first_order_higher_degree.

    Returns the same result structure as other solvers:
      {"solution_summary": <latex str>, "steps": [ {rule_name, result, explanation}, ... ] }
    """
    try:
        steps = []
        p = sympy.Symbol('p')

        if parsed is None:
            raise ValueError("Parsed info required for first-order higher-degree solver.")

        # --------------------------------------------------------------------------------
        # CLAIRAUT case: y = x*p + f(p)
        # --------------------------------------------------------------------------------
        if parsed['type'] == 'clairaut':
            f_p = sympy.simplify(parsed['f'])

            # Step 1: show Clairaut form
            expr_clairaut = sympy.Eq(y, x * p + f_p)
            steps.append({
                "rule_name": "Detect Clairaut Form",
                "result": sympy.latex(expr_clairaut),
                "explanation": "Rewrite the equation using $p = \\dfrac{dy}{dx}$."
            })

            # Step 2: differentiate w.r.t x -> (x + f'(p)) dp/dx = 0
            fprime = sympy.diff(f_p, p)
            steps.append({
                "rule_name": "Differentiate",
                "result": r"\bigl(x + " + sympy.latex(fprime) + r"\bigr)\,\frac{dp}{dx} = 0",
                "explanation": r"Differentiating $y = x p + f(p)$ and using $p = \dfrac{dy}{dx}$."
            })

            # Step 3: dp/dx = 0 => p = C (general family)
            C = sympy.Symbol('C')
            y_gen = sympy.simplify(C * x + f_p.subs(p, C))
            steps.append({
                "rule_name": "General Family (Clairaut)",
                "result": sympy.latex(sympy.Eq(y, y_gen)),
                "explanation": r"From $\dfrac{dp}{dx} = 0$ we get $p = C$; substitute back to obtain the general family."
            })

            # Step 4: singular envelope: solve x + f'(p) = 0 for p (if possible)
            eq_for_p = sympy.Eq(fprime, -x)
            try:
                p_solutions = sympy.solve(eq_for_p, p, dict=False)
            except Exception:
                p_solutions = []

            if p_solutions:
                for p_s in p_solutions:
                    p_s_simpl = sympy.simplify(p_s)
                    y_s = sympy.simplify(x * p_s_simpl + f_p.subs(p, p_s_simpl))
                    steps.append({
                        "rule_name": "Singular Condition",
                        "result": sympy.latex(sympy.Eq(sympy.Symbol('p'), p_s_simpl)),
                        "explanation": r"Solve $f'(p) = -x$ for $p$, producing the envelope slope."
                    })
                    steps.append({
                        "rule_name": "Singular (Envelope) Solution",
                        "result": sympy.latex(sympy.Eq(y, y_s)),
                        "explanation": r"Substitute $p(x)$ into $y = x p + f(p)$ to obtain the singular solution (envelope)."
                    })

                # summary: general family + singular
                family_ltx = sympy.latex(sympy.Eq(y, y_gen))
                singular_ltx = " \\quad ".join(sympy.latex(sympy.Eq(y, sympy.simplify(x * sympy.simplify(p_s) + f_p.subs(p, sympy.simplify(p_s)))) ) for p_s in p_solutions)
                summary_ltx = family_ltx + r" \qquad \text{Singular: } " + singular_ltx
            else:
                steps.append({
                    "rule_name": "Singular Condition (implicit)",
                    "result": sympy.latex(eq_for_p),
                    "explanation": r"Could not solve explicitly for $p$; singular envelope is given implicitly by this relation and $y = x p + f(p)$."
                })
                summary_ltx = sympy.latex(sympy.Eq(y, y_gen)) + r" \qquad \text{Singular: solve } " + sympy.latex(eq_for_p)

            return {"solution_summary": summary_ltx, "steps": steps}

        # --------------------------------------------------------------------------------
        # SOLVABLE_FOR_P case: algebraic equation F(x,y,p) = 0, degree>=2 in p
        # --------------------------------------------------------------------------------
        elif parsed['type'] == 'solvable_for_p':
            p_branches = parsed.get('p_solutions', [])
            deg_p = parsed.get('deg_p', None)

            # Step: show algebraic equation in p (substitution)
            # Build F for display
            F_display = sympy.simplify((eq.lhs - eq.rhs).subs(sympy.Derivative(y, x), p))
            steps.append({
                "rule_name": "Algebraic equation in p",
                "result": sympy.latex(sympy.Eq(F_display, 0)),
                "explanation": r"Replace $p = \dfrac{dy}{dx}$ to get an algebraic equation $F(x,y,p)=0$ to solve for $p$."
            })

            # Show degree if known
            if deg_p is not None:
                steps.append({
                    "rule_name": "Degree in p",
                    "result": f"degree(p) = {deg_p}",
                    "explanation": "This equation is nonlinear in $p$ (degree ≥ 2), so multiple algebraic branches for p may exist."
                })

            # Step: list branches for p
            if not p_branches:
                steps.append({
                    "rule_name": "Solve for p (no explicit branches)",
                    "result": "No explicit algebraic branches found for p.",
                    "explanation": "Unable to obtain closed-form algebraic branches for p."
                })
                return {"solution_summary": sympy.latex(eq), "steps": steps}

            pretty_branches = []
            for i, pb in enumerate(p_branches, start=1):
                pb_simpl = sympy.simplify(pb)
                pretty_branches.append(pb_simpl)
                steps.append({
                    "rule_name": f"Branch {i}: algebraic root",
                    "result": sympy.latex(sympy.Eq(p, pb_simpl)),
                    "explanation": "One algebraic branch for p found from the polynomial/algebraic equation."
                })

            # For each branch reduce to ODE y' = phi(x,y) and attempt to solve with more detail
            branch_summaries = []
            for i, pb in enumerate(pretty_branches, start=1):
                # if pb contains plain symbol 'y', replace with y(x) for correct ODE formation
                try:
                    pb_replaced = pb.subs(sympy.Symbol('y'), y)
                except Exception:
                    pb_replaced = pb

                ode = sympy.Eq(sympy.Derivative(y, x), pb_replaced)
                steps.append({
                    "rule_name": f"Branch {i}: Reduced ODE",
                    "result": sympy.latex(ode),
                    "explanation": "Replace p by this branch to get a first-order ODE to solve."
                })

                # If pb_replaced does not contain y (i.e. depends only on x) we can integrate directly
                pb_free_symbols = set(pb_replaced.free_symbols)
                # detect presence of y by checking for sympy.Function 'y' in expression
                depends_on_y = False
                for atom in pb_replaced.atoms():
                    if isinstance(atom, sympy.Function) and atom.func == sympy.Function('y'):
                        depends_on_y = True
                        break
                # simpler detection: if pb contains Symbol('y') then depends on y
                if pb_replaced.has(sympy.Symbol('y')) or depends_on_y:
                    # dependent on y: try dsolve
                    try:
                        sol = sympy.dsolve(ode, y)
                        if sol is not None:
                            sol_ltx = sympy.latex(sol)
                            steps.append({
                                "rule_name": f"Branch {i}: Solution (automatic)",
                                "result": sol_ltx,
                                "explanation": "Solution obtained by the automatic solver for this branch (may be implicit)."
                            })
                            branch_summaries.append(sol_ltx)
                        else:
                            steps.append({
                                "rule_name": f"Branch {i}: Solution (none found)",
                                "result": sympy.latex(ode),
                                "explanation": "Automatic solver returned no solution; the reduced ODE is shown."
                            })
                            branch_summaries.append(sympy.latex(ode))
                    except Exception as e:
                        steps.append({
                            "rule_name": f"Branch {i}: Solver error",
                            "result": sympy.latex(ode),
                            "explanation": f"Automatic solving failed: {str(e)}"
                        })
                        branch_summaries.append(sympy.latex(ode))
                else:
                    # pb_replaced depends only on x -> integrate directly: y = ∫ pb dx + C
                    try:
                        integral = sympy.integrate(pb_replaced, x)
                        y_solution = sympy.Eq(y, integral + sympy.Symbol('C'))
                        steps.append({
                            "rule_name": f"Branch {i}: Quadrature (direct integration)",
                            "result": sympy.latex(sympy.Eq(sympy.Derivative(y, x), pb_replaced)),
                            "explanation": "The branch gives y' = f(x). Integrate both sides to obtain y."
                        })
                        steps.append({
                            "rule_name": f"Branch {i}: Integral",
                            "result": sympy.latex(sympy.Eq(y, integral + sympy.Symbol('C'))),
                            "explanation": "Integrated the right-hand side with respect to x."
                        })
                        branch_summaries.append(sympy.latex(y_solution))
                    except Exception:
                        # fallback to dsolve
                        try:
                            sol = sympy.dsolve(ode, y)
                            if sol is not None:
                                steps.append({
                                    "rule_name": f"Branch {i}: Solution (automatic fallback)",
                                    "result": sympy.latex(sol),
                                    "explanation": "Automatic solver used as fallback."
                                })
                                branch_summaries.append(sympy.latex(sol))
                            else:
                                steps.append({
                                    "rule_name": f"Branch {i}: No closed-form solution",
                                    "result": sympy.latex(ode),
                                    "explanation": "Could not integrate or find closed-form solution for this branch."
                                })
                                branch_summaries.append(sympy.latex(ode))
                        except Exception as e:
                            steps.append({
                                "rule_name": f"Branch {i}: Solver error (fallback)",
                                "result": sympy.latex(ode),
                                "explanation": f"Automatic solving failed in fallback: {str(e)}"
                            })
                            branch_summaries.append(sympy.latex(ode))

            summary_ltx = r"\text{Branches solved: } " + " \\; ".join(branch_summaries)
            return {"solution_summary": summary_ltx, "steps": steps}

        else:
            raise ValueError("Unknown parsed type for first-order higher-degree solver.")
    except Exception as e:
        traceback.print_exc()
        return {"error": f"First-order higher-degree solver failed: {str(e)}"}

# --- Euler (Cauchy–Euler) detector & stepwise solver ---
# REPLACE: _is_euler
def _is_euler(eq, y, x):
    """
    Detect Cauchy–Euler equations: coefficients must be (constant) * x**k for each derivative order k.
    Returns the detected order (int) if Euler-type; else 0.
    """
    try:
        rearr = (eq.lhs - eq.rhs).expand()
        # find highest derivative order present (up to a reasonable limit)
        max_order = 0
        for i in range(1, 13):
            if rearr.has(y.diff(x, i)):
                max_order = max(max_order, i)
        if max_order < 1:
            return 0

        # For Euler: coefficient of y^(k) should be const * x**k (or zero).
        for k in range(0, max_order + 1):
            coeff = sympy.simplify(rearr.coeff(y.diff(x, k)))
            if coeff == 0:
                continue
            # Try to factor out x**k
            if k == 0:
                candidate = sympy.simplify(coeff)
            else:
                candidate = sympy.simplify(sympy.simplify(coeff / (x**k)))
            # candidate must not contain x (i.e., a genuine constant w.r.t x)
            if candidate.has(x):
                return 0
        return max_order
    except Exception:
        return 0

# REPLACE: _solve_euler
def _solve_euler(eq, y, x, order):
    """
    Solve homogeneous Euler (Cauchy–Euler) equation stepwise.

    This function:
      - extracts coefficients of x^k * y^{(k)} and enforces they are constant* x^k,
      - builds the indicial polynomial P(r) = sum_k a_k * r*(r-1)*...*(r-k+1),
      - solves P(r)=0 and assembles complementary function handling distinct, repeated and complex roots.
    For non-homogeneous Euler equations we conservatively fall back to SymPy's solver (no risky transforms here).
    """
    steps = []
    try:
        # Collect coefficients a_k where original term is a_k * x**k * y^{(k)}
        rearr = (eq.lhs - eq.rhs).expand()
        a_consts = []  # a_order, ..., a_0
        for k in range(order, -1, -1):
            coeff = sympy.simplify(rearr.coeff(y.diff(x, k)))
            if coeff == 0:
                a_k = sympy.Integer(0)
            else:
                if k == 0:
                    a_k = sympy.simplify(coeff)
                else:
                    a_k = sympy.simplify(sympy.simplify(coeff / (x**k)))
                # safety: if a_k still depends on x then this isn't a pure Euler equation
                if a_k.has(x):
                    # give up and fallback
                    return {"error": "Detected coefficients depend on x; not a pure Euler equation."}
            a_consts.append(sympy.simplify(a_k))

        # Build indicial polynomial using falling factorials:
        r = sympy.Symbol('r')
        indicial = sympy.Integer(0)
        # a_consts has order -> a_n, ..., a_0
        for i, a_k in enumerate(a_consts):
            k = order - i  # derivative order
            if k == 0:
                factor = sympy.Integer(1)
            else:
                factor = sympy.Integer(1)
                for j in range(k):
                    factor = sympy.expand(factor * (r - j))
            indicial += sympy.expand(a_k * factor)

        indicial = sympy.simplify(indicial)
        steps.append({"rule_name": "Indicial Polynomial", "result": sympy.latex(sympy.Eq(indicial, 0)),
                      "explanation": "Substitute trial y = x^r and collect constant coefficients to form the indicial polynomial."})

        # Solve indicial polynomial
        # Use sympy.roots for multiplicities
        try:
            roots_dict = sympy.roots(sympy.simplify(indicial), r)
        except Exception:
            # fallback to solve if roots fails
            roots_solutions = sympy.solve(sympy.Eq(indicial, 0), r)
            roots_dict = {rt: sympy.Integer(1) for rt in roots_solutions}

        factored = sympy.factor(sympy.simplify(indicial))
        steps.append({"rule_name": "Factor / Roots (indicial)", "result": sympy.latex(factored),
                      "explanation": f"Indicial roots (with multiplicity): {', '.join(f'{sympy.latex(k)}: {v}' for k,v in roots_dict.items())}"})

        # Build complementary function
        x_sym = sympy.Symbol('x')
        CF_terms = []
        C_counter = 1
        handled = set()
        for root, mult in roots_dict.items():
            root_s = sympy.simplify(root)
            # if root is a numeric approximation, keep sympy.simplify
            if root_s in handled:
                continue
            if sympy.im(root_s) == 0:
                # real root
                for m in range(int(mult)):
                    Ci = sympy.Symbol(f'C{C_counter}'); C_counter += 1
                    if m == 0:
                        CF_terms.append(Ci * x_sym**sympy.simplify(root_s))
                    else:
                        CF_terms.append(Ci * x_sym**sympy.simplify(root_s) * sympy.log(x_sym)**m)
                handled.add(root_s)
            else:
                # complex root a +/- i b -> produce cos/sin pair multiplied by x^a, include log factors for multiplicity
                a = sympy.simplify(sympy.re(root_s))
                b = sympy.simplify(sympy.im(root_s))
                for m in range(int(mult)):
                    A = sympy.Symbol(f'C{C_counter}'); C_counter += 1
                    B = sympy.Symbol(f'C{C_counter}'); C_counter += 1
                    term = x_sym**a * (A * sympy.cos(b * sympy.log(x_sym)) + B * sympy.sin(b * sympy.log(x_sym)))
                    if m > 0:
                        term = term * sympy.log(x_sym)**m
                    CF_terms.append(term)
                handled.add(root_s)
                handled.add(sympy.conjugate(root_s))

        CF = sympy.Add(*CF_terms) if CF_terms else sympy.Integer(0)
        steps.append({"rule_name": "Complementary Function (Euler)", "result": sympy.latex(sympy.Eq(y, CF)),
                      "explanation": "Construct complementary function from indicial roots (use log factors for repeated roots; cos/sin(log) for complex roots)."})

        # If homogeneous, return final
        rhs = sympy.simplify(eq.rhs)
        if rhs == 0:
            final = sympy.Eq(y, CF)
            steps.append({"rule_name": "Final Solution (Homogeneous Euler)", "result": sympy.latex(final), "explanation": ""})
            return {"solution_summary": sympy.latex(final), "steps": steps}

        # Non-homogeneous Euler: conservative fallback to SymPy's dsolve (avoid brittle transforms here)
        steps.append({"rule_name": "Non-homogeneous Euler", "result": "Fallback", "explanation": "Non-homogeneous Euler handled via SymPy fallback to avoid symbolic transform errors."})
        try:
            sol = sympy.dsolve(eq, y)
            steps.append({"rule_name": "General Solution (fallback)", "result": sympy.latex(sol), "explanation": "Solution via SymPy fallback for the non-homogeneous Euler equation."})
            return {"solution_summary": sympy.latex(sol), "steps": steps}
        except Exception as e:
            steps.append({"rule_name": "Fallback failure", "result": str(e), "explanation": "SymPy fallback failed to solve non-homogeneous Euler."})
            return {"solution_summary": sympy.latex(sympy.Eq(y, CF)), "steps": steps}

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Euler solver failed: {str(e)}"}


# --- 4. The New Main Solver (fixed parsing of y and derivatives) ---
def solve_differential_equation(expression_str: str):
    """Classifies and solves a differential equation, providing steps.

    Robust parsing of prime-notation derivatives (y', y'', y''') and plain "y"
    into callable form y(x) before sympifying. First-order higher-degree
    detection (Clairaut / solvable-for-p) is tested *before* the linear test.
    """
    try:
        x, C = sympy.symbols('x C')
        y_func = sympy.Function('y')
        y = y_func(x)

        normalized_expr = normalize_expression(expression_str)

        if normalized_expr.count('=') != 1:
            return {"error": "Input must contain exactly one '='."}

        # Work on a local copy for parsing
        parsed_str = normalized_expr

        # 1) Prefer explicit Leibniz forms first to avoid prime-clashes
        parsed_str = parsed_str.replace('dy/dx', 'Derivative(y(x), x)')
        parsed_str = parsed_str.replace('d2y/dx2', 'Derivative(y(x), (x, 2))')
        parsed_str = parsed_str.replace('d3y/dx3', 'Derivative(y(x), (x, 3))')

        # 2) Replace prime-notation derivatives like y''' y'' y'
        def _prime_to_derivative(m):
            primes = m.group(1)
            n = len(primes)
            if n == 1:
                return 'Derivative(y(x), x)'
            else:
                return f'Derivative(y(x), (x, {n}))'

        parsed_str = re.sub(r"y('+)", _prime_to_derivative, parsed_str)

        # 3) Insert explicit multiplication where appropriate (e.g., 2y -> 2*y)
        parsed_str = re.sub(r'(\d)\s*(?=[A-Za-z(])', r'\1*', parsed_str)

        # 4) After derivatives are expanded, convert standalone 'y' tokens to 'y(x)'
        #    but do not replace occurrences that are already y(...)
        parsed_str = re.sub(r'(?<![A-Za-z0-9_])y(?!\s*\()', 'y(x)', parsed_str)

        # 5) Normalize caret exponent notation to Python **
        parsed_str = parsed_str.replace('^', '**')

        lhs_str, rhs_str = parsed_str.split('=', 1)

        locals_map = {
            'y': y_func, 'x': x,
            # Common math functions
            'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
            'exp': sympy.exp, 'log': sympy.log, 'sqrt': sympy.sqrt,
            'sinh': sympy.sinh, 'cosh': sympy.cosh, 'tanh': sympy.tanh,
            'pi': sympy.pi, 'E': sympy.E
        }

        # build sympy Eq
        equation = sympy.Eq(sympy.sympify(lhs_str, locals=locals_map),
                            sympy.sympify(rhs_str, locals=locals_map))

        # --- CLASSIFICATION CHAIN (priority order) ---
        # 1) First-order higher-degree (Clairaut / solvable-for-p) - priority
        parsed_first_order_hd = None
        try:
            parsed_first_order_hd = _is_first_order_higher_degree(equation, y, x)
        except Exception as e_det:
            # detector should be defensive; log and continue to other checks
            print("Detector _is_first_order_higher_degree raised:", repr(e_det))

        if parsed_first_order_hd:
            print("Classifier: Detected First-Order Higher-Degree Equation. parsed ->", parsed_first_order_hd)
            return _solve_first_order_higher_degree(equation, y, x, parsed_first_order_hd)

        # 2) First-order linear
        try:
            if _is_linear(equation, y, x):
                print("Classifier: Detected First-Order Linear Equation.")
                return _solve_linear_step_by_step(equation, y, x)
        except Exception as e_lin:
            print("Linear classifier raised:", repr(e_lin))
            # continue to other checks

        # 3) Constant-coefficient linear (higher order)
        try:
            order = _is_constant_coeff_linear(equation, y, x)
            if order:
                print(f"Classifier: Detected Constant-Coefficient Linear Equation (order {order}).")
                return _solve_constant_coeff_linear(equation, y, x, order)
        except Exception as e_cc:
            print("Constant-coeff classifier raised:", repr(e_cc))
            # continue to next check

        # 4) Euler (Cauchy–Euler)
        try:
            e_order = _is_euler(equation, y, x)
            if e_order:
                print(f"Classifier: Detected Euler (Cauchy–Euler) Equation (order {e_order}).")
                return _solve_euler(equation, y, x, e_order)
        except Exception as e_eu:
            print("Euler classifier raised:", repr(e_eu))
            # continue to fallback

        # --- FALLBACK: use SymPy general solver ---
        print("Classifier: No specific type matched. Using general solver.")
        try:
            solution = sympy.dsolve(equation, y)
        except Exception as e:
            raise ValueError(f"SymPy failed to solve: {e}")

        if solution is None:
            raise ValueError("SymPy returned no solution.")

        if isinstance(solution, (list, tuple)) and len(solution) == 0:
            raise ValueError("SymPy returned an empty solution.")

        return {
            "solution_summary": sympy.latex(solution),
            "steps": [
                {"rule_name": "Initial Equation", "result": sympy.latex(equation)},
                {"rule_name": "General Solution", "result": sympy.latex(solution)}
            ]
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Failed to process equation: {str(e)}"}
