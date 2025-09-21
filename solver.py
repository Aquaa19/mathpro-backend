# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import re
import sympy

# --- Normalization ---------------------------------------------------------

def normalize_expression(expr: str) -> str:
    """
    Cleans up a user-input string to be more compatible with SymPy using a multi-pass approach.
    - Handles unicode superscripts and sqrt, primes (y', y''), and Leibniz d^n y / dx^n notations.
    - Inserts implicit multiplication signs for common cases.
    - Converts '^' to '**'.
    Returns a string suitable for sympy.sympify / sympy.parse_expr (with appropriate locals).
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a string")

    # 0. Remove extra spaces (but keep spaces needed by regex detection)
    expr = expr.strip()

    # Pass 1: Map common Unicode characters to ASCII forms
    char_map = {
        '²': '^2', '³': '^3', '¹': '^1', '⁰': '^0', '⁴': '^4', '⁵': '^5',
        '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '’': "'", '”': "''", '‘': "'",
        '√': 'sqrt'
    }
    for k, v in char_map.items():
        expr = expr.replace(k, v)

    # Remove stray spaces around operators to make recognition simpler
    expr = re.sub(r'\s+', ' ', expr)
    expr = re.sub(r'\s*([+\-*/=^(),])\s*', r'\1', expr)
    # Now remove remaining leading/trailing spaces
    expr = expr.strip()

    # Pass 2: Replace derivative notations with placeholders safely using re.sub with callback
    deriv_map = {}
    counter = 0

    def new_placeholder():
        nonlocal counter
        ph = f"__DERIV_{counter}__"
        counter += 1
        return ph

    # Pattern A: Leibniz style d^n y / dx^n  (allow optional ^, optional spaces, digits for n)
    # Examples matched: "dy/dx", "d^2y/dx^2", "d 3 y / dx3" (with spaces), but we normalize common forms.
    pattern_leibniz = re.compile(
        r'd\^?\s*(\d+)?\s*y\s*/\s*dx\^?\s*(\d+)?', flags=re.IGNORECASE
    )

    def _replace_leibaut(match):
        n1 = match.group(1)
        n2 = match.group(2)
        # determine order: prefer numeric capture, default to 1
        n = None
        if n1 and n1.isdigit():
            n = int(n1)
        elif n2 and n2.isdigit():
            n = int(n2)
        else:
            n = 1
        ph = new_placeholder()
        deriv_map[ph] = f"Derivative(y(x),(x,{n}))" if n != 1 else "Derivative(y(x),x)"
        return ph

    expr = pattern_leibniz.sub(_replace_leibaut, expr)

    # Pattern B: Prime notation y', y'', y''' (one or more ASCII single quotes)
    # Capture sequences like y', y''', etc. (we assume variable named exactly 'y' — extendable)
    pattern_primes = re.compile(r"y('{1,})")

    def _replace_primes(match):
        quotes = match.group(1)
        n = len(quotes)
        ph = new_placeholder()
        deriv_map[ph] = f"Derivative(y(x),(x,{n}))" if n != 1 else "Derivative(y(x),x)"
        return ph

    expr = pattern_primes.sub(_replace_primes, expr)

    # Pass 3: Insert implicit multiplications
    # We'll treat common functions/identifiers as atomic tokens (prevent injecting * inside them)
    # Allowed function names (you can extend this set as needed)
    func_names = r'(?:sin|cos|tan|log|exp|sqrt|Derivative|__DERIV_\d+__)'
    # Insert between number and var/func/(
    expr = re.sub(r'(?P<num>(?<![A-Za-z_])\d+(?:\.\d+)?))(?P<f>(?:[A-Za-z_]\w*|' + func_names + r'|\())',
                  r'\g<num>*\g<f>', expr)
    # Insert between closing parenthesis or variable and open parenthesis, function, or variable
    # (e.g., ')(', 'x(', 'x y', ')x', ')sin')
    expr = re.sub(r'(?P<left>[A-Za-z0-9_)\]])(?P<right>(?:\(|' + func_names + r'|[A-Za-z_]))',
                  r'\g<left>*\g<right>', expr)

    # Handle derivative placeholders specifically: ensure they are treated atomically with surrounding text
    # e.g., "__DERIV_0__y" -> "__DERIV_0__*y"
    expr = re.sub(r'(__DERIV_\d+__)(?=[A-Za-z_(])', r'\1*', expr)
    expr = re.sub(r'(?<=[A-Za-z0-9_)])(__DERIV_\d+__)', r'*\1', expr)

    # Pass 4: Convert caret to Python exponent
    expr = expr.replace("^", "**")

    # Pass 5: Replace placeholders back to SymPy text
    # Do replacements in a way that avoids accidental partial replacements
    # (longer placeholders first — though our placeholders are uniform)
    for ph, sym_text in deriv_map.items():
        expr = expr.replace(ph, sym_text)

    # Final cleanup: collapse multiple stars from previous heuristics and remove accidental spaces
    expr = re.sub(r'\*\*', '**', expr)
    expr = re.sub(r'\*\*', '**', expr)
    expr = re.sub(r'\s+', '', expr)

    return expr

# --- Step-by-Step Solvers -------------------------------------------------

def _solve_linear_step_by_step(eq, y, x):
    """Generates steps for first-order linear equations."""
    dydx = y.diff(x)
    rearranged = (eq.lhs - eq.rhs).expand()

    # Ensure coefficient of y' is 1
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
        "result": f"I(x) = e^{{\\int P(x) dx}} = {sympy.latex(integrating_factor)}",
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

def _solve_bernoulli_step_by_step(eq, y, x, n):
    """Generates steps for Bernoulli equations: y' + P(x)y = Q(x)y^n."""
    dydx = y.diff(x)
    rearranged = (eq.lhs - eq.rhs).expand()
    p_x = rearranged.coeff(y, 1)
    q_x = -rearranged.coeff(y**n, 1)

    steps = [{
        "rule_name": "Identify Bernoulli Form",
        "result": f"P(x) = {sympy.latex(p_x)}, Q(x) = {sympy.latex(q_x)}, n = {n}",
        "explanation": "The equation is a Bernoulli equation: y' + P(x)y = Q(x)y^n."
    }]
    v = sympy.Function('v')(x)
    subst_power = 1 - n
    steps.append({
        "rule_name": "Substitute v = y^(1-n)",
        "result": f"v = y^{{{subst_power}}}",
        "explanation": "This substitution transforms the equation into a linear form."
    })
    # Form linear equation in v: v' + (1-n) p(x) v = (1-n) q(x)
    linear_v_eq = sympy.Eq(sympy.diff(v, x) + (1 - n) * p_x * v, (1 - n) * q_x)
    steps.append({
        "rule_name": "Transform into a Linear Equation",
        "result": sympy.latex(linear_v_eq),
        "explanation": "The equation becomes a first-order linear equation in v."
    })
    # Solve linear equation for v using dsolve (explicit)
    v_sol = sympy.dsolve(linear_v_eq, v)
    steps.append({
        "rule_name": "Solve the Linear Equation for v",
        "result": sympy.latex(v_sol),
        "explanation": "Solve the new linear equation for v(x) using standard methods."
    })
    # Back-substitute step description
    steps.append({
        "rule_name": "Back-substitute to find y",
        "result": ("Back-substitute v = y^{{{}}}. See v(x) = {}"
                   .format(subst_power, sympy.latex(v_sol.rhs if isinstance(v_sol, sympy.Equality) else v_sol.args[1]))),
        "explanation": f"Substitute v = y^{{{subst_power}}} back to get (implicitly) the solution for y(x)."
    })
    return {"solution_summary": sympy.latex(v_sol), "steps": steps}

def _solve_homogeneous_step_by_step(eq, y, x):
    """Generates steps for homogeneous equations (first-order)."""
    dydx = y.diff(x)
    f_xy = eq.rhs
    v = sympy.Function('v')(x)
    steps = [{
        "rule_name": "Verify Homogeneity",
        "result": "f(tx, ty) = f(x, y)",
        "explanation": "The equation is homogeneous."
    }]
    # Substitute y = v*x and dy/dx = v + x*dv/dx
    subst_eq = sympy.Eq(v + x * sympy.diff(v, x), f_xy.subs(y, v * x))
    steps.append({
        "rule_name": "Substitute y = vx",
        "result": sympy.latex(subst_eq),
        "explanation": "Substitute y=vx and dy/dx = v + x(dv/dx) to make the equation separable."
    })
    separable_form = sympy.Eq(sympy.diff(v, x), sympy.solve(subst_eq, sympy.diff(v, x))[0])
    steps.append({
        "rule_name": "Separate Variables",
        "result": sympy.latex(separable_form),
        "explanation": "Rearrange to separate the v and x variables."
    })
    # Integrate both sides
    integral_v = sympy.integrate(1 / (separable_form.rhs - v), v)
    integral_x = sympy.integrate(1 / x, x)
    integrated_eq = sympy.Eq(integral_v, integral_x + sympy.Symbol('C'))
    steps.append({
        "rule_name": "Integrate Both Sides",
        "result": sympy.latex(integrated_eq),
        "explanation": "Integrate both sides of the separated equation."
    })
    final_solution = integrated_eq.subs(v, y / x)
    steps.append({
        "rule_name": "Back-substitute v = y/x",
        "result": sympy.latex(final_solution),
        "explanation": "Substitute back to get the final implicit solution."
    })
    return {"solution_summary": sympy.latex(final_solution), "steps": steps}

def _solve_exact_step_by_step(M, N, y, x):
    """Generates steps for exact equations."""
    steps = []
    dM_dy = sympy.diff(M, y)
    dN_dx = sympy.diff(N, x)
    steps.append({
        "rule_name": "Verify Exactness (∂M/∂y = ∂N/∂x)",
        "result": f"{sympy.latex(dM_dy)} = {sympy.latex(dN_dx)}",
        "explanation": "The equation is exact because the partial derivatives are equal."
    })
    integral_M = sympy.integrate(M, x)
    steps.append({
        "rule_name": "Integrate M(x,y) dx",
        "result": f"f(x,y) = \\int M(x,y) dx = {sympy.latex(integral_M)} + g(y)",
        "explanation": "Integrate M w.r.t. x, adding a function g(y)."
    })
    df_dy = sympy.diff(integral_M, y)
    g_prime_y = sympy.Eq(df_dy + sympy.Symbol("g'(y)"), N)
    steps.append({
        "rule_name": "Differentiate w.r.t. y and equate to N",
        "result": sympy.latex(g_prime_y),
        "explanation": "Differentiate the result w.r.t. y and set equal to N(x,y) to find g'(y)."
    })
    g_prime_val = sympy.solve(g_prime_y, sympy.Symbol("g'(y)"))[0]
    g_y = sympy.integrate(g_prime_val, y)
    steps.append({
        "rule_name": "Find g(y)",
        "result": f"g(y) = \\int {sympy.latex(g_prime_val)} dy = {sympy.latex(g_y)}",
        "explanation": "Integrate g'(y) to find g(y)."
    })
    final_solution = sympy.Eq(integral_M + g_y, sympy.Symbol('C'))
    steps.append({
        "rule_name": "Final Implicit Solution",
        "result": sympy.latex(final_solution),
        "explanation": "The final solution is given by f(x,y) = C."
    })
    return {"solution_summary": sympy.latex(final_solution), "steps": steps}

def _solve_clairaut_step_by_step(eq, y, x, p):
    """Generates steps for Clairaut's equations: y = xp + f(p)."""
    C = sympy.Symbol('C')
    # Solve for y explicitly and isolate f(p) = y - x p
    y_solved = sympy.solve(eq, y)
    if not y_solved:
        return {"error": "Can't rearrange to Clairaut's form."}
    y_expr = y_solved[0]
    f_p = (y_expr - x * p).simplify()
    steps = [{
        "rule_name": "Identify Clairaut's Form",
        "result": f"f(p) = {sympy.latex(f_p)}",
        "explanation": "The equation is in Clairaut's form y = xp + f(p), where p = dy/dx."
    }]
    df_dp = sympy.diff(f_p, p)
    diff_eq_latex = f"\\left( x + {sympy.latex(df_dp)} \\right) \\frac{{dp}}{{dx}} = 0"
    steps.append({
        "rule_name": "Differentiate w.r.t. x",
        "result": diff_eq_latex,
        "explanation": "Differentiating the entire equation w.r.t. x yields two cases."
    })
    general_solution = eq.subs(p, C)
    steps.append({
        "rule_name": "Case 1: Find General Solution",
        "result": sympy.latex(general_solution),
        "explanation": "From dp/dx = 0, we get p = C. Substituting this back gives the general solution."
    })
    p_from_x = sympy.solve(x + df_dp, p)
    singular_solution = None
    if p_from_x:
        singular_solution = sympy.Eq(y, (x * p + f_p).subs(p, p_from_x[0]))
        steps.append({
            "rule_name": "Case 2: Find Singular Solution",
            "result": sympy.latex(singular_solution),
            "explanation": "From x + f'(p) = 0, we solve for p and substitute back to find the singular solution."
        })
    summary = f"General: {sympy.latex(general_solution)}; Singular: {sympy.latex(singular_solution) if singular_solution is not None else 'None'}"
    return {"solution_summary": summary, "steps": steps}


# --- REPLACED: linear constant-coefficient solver (with PI by undetermined coefficients) ---

def _solve_linear_constant_coeff_step_by_step(eq, y, x):
    """
    Solves n-th order linear ODEs with constant coefficients using the
    Method of Undetermined Coefficients for the particular integral.
    """
    steps = []
    C_symbols = [sympy.Symbol(f'C{i+1}') for i in range(10)]

    # --- Part 1: Find the Complementary Function (CF) ---
    lhs = eq.lhs
    rhs = eq.rhs

    # Build auxiliary polynomial robustly by collecting coefficients for y, y', y'', ...
    m = sympy.Symbol('m')
    coeffs = {}
    max_order = 0
    for n in range(0, 21):  # check up to 20th order
        deriv = y.diff(x, n) if n > 0 else y
        c = sympy.simplify(lhs.coeff(deriv, 1))
        if c != 0:
            # ensure coefficient is constant (no x or y symbols)
            if c.free_symbols:
                # not constant coefficient; abort specialized solver
                raise ValueError("Non-constant coefficient found; this solver expects constant coefficients.")
            coeffs[n] = sympy.simplify(c)
            max_order = max(max_order, n)

    if max_order == 0 and rhs != 0:
        # No derivative terms found -> not a linear ODE in expected form
        raise ValueError("No derivative terms found; cannot form auxiliary equation.")

    # form polynomial a_n m^n + ... + a_0
    aux_eq_poly = sum(coeffs.get(n, 0) * m**n for n in range(0, max_order + 1))
    aux_eq = sympy.Eq(aux_eq_poly, 0)
    steps.append({
        "rule_name": "Step 1: Find the Complementary Function (CF)",
        "result": "First, solve the homogeneous equation: " + sympy.latex(sympy.Eq(lhs, 0)),
        "explanation": "The general solution is y = y_c + y_p, where y_c is the Complementary Function and y_p is the Particular Integral."
    })
    steps.append({
        "rule_name": "Form the Auxiliary Equation",
        "result": sympy.latex(aux_eq),
        "explanation": "Substitute y = e^(m x) into the homogeneous equation."
    })

    # Get roots (with multiplicity)
    poly_m = sympy.Poly(aux_eq_poly, m)
    roots = sympy.roots(poly_m)
    steps.append({
        "rule_name": "Find the Roots",
        "result": f"m = {sympy.latex(list(roots.keys()))}",
        "explanation": "Solve the polynomial for its roots."
    })

    # Construct complementary function
    cf_terms = []
    root_index = 0
    for root, multiplicity in roots.items():
        root_s = sympy.simplify(root)
        if sympy.im(root_s) == 0:  # Real root
            for i in range(multiplicity):
                cf_terms.append(C_symbols[root_index] * (x**i) * sympy.exp(root_s * x))
                root_index += 1
        else:  # Complex roots (handle conjugate pairs)
            alpha = sympy.re(root_s)
            beta = sympy.im(root_s)
            # For each multiplicity, we produce cos and sin terms multiplied by x^k
            for i in range(multiplicity):
                term = sympy.exp(alpha * x) * (C_symbols[root_index] * sympy.cos(beta * x) + C_symbols[root_index + 1] * sympy.sin(beta * x)) * (x**i)
                cf_terms.append(term)
                root_index += 2

    complementary_function = sum(cf_terms) if cf_terms else sympy.Integer(0)
    steps.append({
        "rule_name": "Construct the Complementary Function",
        "result": f"y_c = {sympy.latex(complementary_function)}",
        "explanation": "Construct the CF from the roots of the auxiliary equation."
    })

    # --- Part 2: Find the Particular Integral (PI) ---
    if rhs == 0:
        final_solution = sympy.Eq(y, complementary_function)
        steps.append({
            "rule_name": "General Solution (Homogeneous)",
            "result": sympy.latex(final_solution),
            "explanation": "For a homogeneous equation, the general solution is the complementary function."
        })
        return {"solution_summary": sympy.latex(final_solution), "steps": steps}

    # Non-homogeneous case: Method of Undetermined Coefficients (basic support)
    steps.append({
        "rule_name": "Step 2: Find the Particular Integral (PI)",
        "result": "Now, find a particular solution y_p for the non-homogeneous equation.",
        "explanation": "We will use the Method of Undetermined Coefficients based on the form of the right-hand side, f(x)."
    })

    # Guess the form of the PI (simple heuristics: polynomial, exp, trig, or product)
    pi_guess = sympy.sympify(0)

    # Convert rhs to a simplified form for pattern checking
    rhs_simp = sympy.simplify(rhs)

    # Polynomial
    if rhs_simp.is_polynomial(x):
        degree = sympy.degree(rhs_simp, x)
        coeffs_A = [sympy.Symbol(f'A{i}') for i in range(degree + 1)]
        pi_guess = sum(coeffs_A[i] * x**i for i in range(degree + 1))

    # Exponential term containing exp(k*x)
    elif any([func.has(sympy.exp) for func in sympy.preorder_traversal(rhs_simp)]):
        # Attempt to extract exp(k*x) factor; fallback to a single coefficient times rhs
        exp_factors = [node for node in sympy.preorder_traversal(rhs_simp) if node.func == sympy.exp]
        if exp_factors:
            exp_term = exp_factors[0]
            A = sympy.Symbol('A')
            # if exp_term is e^{k*x}, guess A*e^{k*x}
            pi_guess = A * exp_term
        else:
            pi_guess = sympy.Symbol('A') * rhs_simp

    # Trigonometric sin/cos
    elif any([isinstance(node, sympy.sin) or isinstance(node, sympy.cos) for node in sympy.preorder_traversal(rhs_simp)]):
        trig_nodes = [node for node in sympy.preorder_traversal(rhs_simp) if isinstance(node, (sympy.sin, sympy.cos))]
        trig = trig_nodes[0]
        # argument should be k*x typically
        arg = trig.args[0]
        k = arg.coeff(x) if arg.has(x) else sympy.Integer(1)
        A, B = sympy.symbols('A B')
        pi_guess = A * sympy.cos(k * x) + B * sympy.sin(k * x)

    else:
        # Fallback: use undetermined coefficient multiplying the RHS directly
        A = sympy.Symbol('A')
        pi_guess = A * rhs_simp

    steps.append({
        "rule_name": "Guess the Form of y_p",
        "result": f"y_p = {sympy.latex(pi_guess)}",
        "explanation": f"Based on f(x) = {sympy.latex(rhs_simp)}, we guess the form of the particular solution."
    })

    # Check for duplication of any term in CF; if duplicated, multiply guess by x^s until independent
    def _term_equivalent_to_cf_term(pi_term, cf_term):
        # Rough heuristic: compare functional structure after replacing constants with 1
        symbols_in_pi = list(pi_term.atoms(sympy.Symbol))
        subs_map = {s: 1 for s in symbols_in_pi}
        try:
            simplified_ratio = sympy.simplify((pi_term.subs(subs_map)) / (cf_term.subs({c: 1 for c in C_symbols})))
            return simplified_ratio.is_constant
        except Exception:
            return False

    # If pi_guess is a sum, inspect each summand
    multiplied = False
    for cf_t in cf_terms:
        # try to detect duplication; we apply a simpler check: if their functional forms share the same base (exp/cos/sin/polynomial)
        try:
            # if pi_guess and cf_t share same exponential/trig base, we consider duplication
            if (sympy.simplify(sympy.exp(0) * pi_guess).has(sympy.exp) and cf_t.has(sympy.exp)) or \
               (pi_guess.has(sympy.cos) or pi_guess.has(sympy.sin)) and (cf_t.has(sympy.cos) or cf_t.has(sympy.sin)) or \
               (pi_guess.is_polynomial(x) and cf_t.is_polynomial(x)):
                pi_guess = pi_guess * x
                multiplied = True
                steps.append({
                    "rule_name": "Modify Guess for Duplication",
                    "result": f"y_p = {sympy.latex(pi_guess)}",
                    "explanation": "The initial guess duplicates a term in the CF. Multiply the guess by x."
                })
                break
        except Exception:
            continue

    # Solve for the coefficients by substituting y = pi_guess into the LHS and equating to RHS
    # Build operator on y: compute L[y_p] where L is the linear operator represented by lhs
    # To evaluate, we create an expression: lhs_expr.subs(y, pi_guess) but lhs may contain derivatives
    # We'll replace occurrences of y.diff(x, n) with pi_guess.diff(x, n)
    substituted_lhs = lhs
    for n in range(0, max_order + 1):
        deriv = y.diff(x, n) if n > 0 else y
        substituted_lhs = substituted_lhs.subs(deriv, sympy.diff(pi_guess, x, n))

    # Form equation for coefficients
    eq_for_coeffs = sympy.simplify(substituted_lhs - rhs_simp)
    # Unknown symbols in pi_guess
    undetermined_coeffs = sorted(list(pi_guess.free_symbols), key=lambda s: str(s))
    # Remove x and known constants from undetermined set if present
    undetermined_coeffs = [s for s in undetermined_coeffs if s != x and not str(s).startswith('C')]

    solved_coeffs = {}
    if undetermined_coeffs:
        # Convert eq_for_coeffs into a polynomial (in x) or equate coefficients by expanding trig/exponent if possible.
        try:
            # Attempt to collect coefficients by matching powers of x
            # Build equations by equating coefficients of linearly independent functions if possible
            # Use sympy.solve for the system
            # For robust handling, sample enough points and solve linear system for unknowns (works for linear undetermined coefficient problems)
            vars_symbols = undetermined_coeffs
            # Create linear equations by comparing coefficients for a basis of functions (monomials, sin/cos, exp)
            # Fallback: solve by matching coefficients as equations by expanding series at enough points
            # We'll try to expand eq_for_coeffs and equate to zero by collecting coefficients for x powers
            equations = []
            # Try matching coefficients of x up to degree 10
            for k in range(0, 11):
                coeff_k = sympy.simplify(sympy.expand(eq_for_coeffs).coeff(x, k))
                equations.append(sympy.Eq(coeff_k, 0))
            # Also include constant equation when k=0 already included
            # Solve the linear system
            solved_coeffs = sympy.solve(equations, vars_symbols, dict=True)
            if isinstance(solved_coeffs, list) and solved_coeffs:
                solved_coeffs = solved_coeffs[0]
            elif solved_coeffs is None:
                solved_coeffs = {}
        except Exception:
            # Fallback to direct solve of eq_for_coeffs (may work for simple exponential/trig forms)
            try:
                solved_coeffs = sympy.solve(eq_for_coeffs, undetermined_coeffs, dict=True)
                if isinstance(solved_coeffs, list) and solved_coeffs:
                    solved_coeffs = solved_coeffs[0]
            except Exception:
                solved_coeffs = {}
    else:
        # No undetermined coefficients (rare): check if substitution satisfies equation
        if sympy.simplify(eq_for_coeffs) != 0:
            raise ValueError("Failed to find a particular integral using the guessed form.")
        solved_coeffs = {}

    particular_integral = sympy.simplify(pi_guess.subs(solved_coeffs)) if solved_coeffs else sympy.simplify(pi_guess)
    steps.append({
        "rule_name": "Solve for Coefficients",
        "result": f"y_p = {sympy.latex(particular_integral)}",
        "explanation": f"Substitute y_p into the original equation and solve for the unknown coefficients. Found {sympy.latex(solved_coeffs)}."
    })

    # --- Part 3: Combine for General Solution ---
    final_solution = sympy.Eq(y, complementary_function + particular_integral)
    steps.append({
        "rule_name": "Step 3: Construct the General Solution (y = y_c + y_p)",
        "result": sympy.latex(final_solution),
        "explanation": "The full general solution is the sum of the complementary function and the particular integral."
    })

    return {"solution_summary": sympy.latex(final_solution), "steps": steps}


# --- REPLACED: helper detection function ----------------------------------

def _is_linear_constant_coeff(eq, y):
    """Checks if an equation is a linear ODE with constant coefficients."""
    try:
        rearranged = (eq.lhs - eq.rhs).expand()
        # Quick test: RHS must be 0 or a purely function of x (non-homog allowed)
        # We'll allow non-homogeneous since our solver handles both; here we only check coefficients.
        # Collect coefficients for derivative orders
        for n in range(0, 21):
            deriv = y.diff(sympy.Symbol('x'), n) if n > 0 else y
            c = rearranged.coeff(deriv, 1)
            if c != 0:
                # if coefficient depends on x or y, it's not constant-coeff
                if c.free_symbols:
                    return False
        # If we reached here, no non-constant coefficient found (or zero equation) => likely constant coefficient
        return True
    except Exception:
        return False


# --- Helper: linear first-order check -------------------------------------

def _is_linear(eq, y, x):
    """Helper to check for first-order linearity y' + P(x) y = Q(x)."""
    try:
        rearranged = (eq.lhs - eq.rhs).expand()
        dydx = y.diff(x)
        if not rearranged.has(dydx):
            return False
        # collect in terms of y and y'
        collected = sympy.collect(rearranged, [y, dydx])
        coeff_dydx = collected.coeff(dydx, 1)
        if coeff_dydx == 0:
            return False
        collected = collected / coeff_dydx
        p_x = collected.coeff(y, 1)
        q_x = -(collected - dydx - p_x * y)
        return not p_x.has(y) and not q_x.has(y)
    except Exception:
        return False

# --- Main Solver and Classifier -------------------------------------------

def solve_differential_equation(expression_str: str):
    """Classifies and solves a differential equation, providing steps."""
    try:
        x, C = sympy.symbols('x C')
        y = sympy.Function('y')(x)
        dydx = y.diff(x)
        p = sympy.Symbol('p')

        normalized_expr = normalize_expression(expression_str)

        # Check for Clairaut's form first
        try:
            # replace y' with symbol p for detection
            temp = normalized_expr.replace(str(dydx), 'p')
            temp_lhs, temp_rhs = temp.split('=')
            locals_map_clairaut = {'y': y, 'x': x, 'p': p}
            temp_eq = sympy.Eq(sympy.sympify(temp_lhs, locals=locals_map_clairaut),
                               sympy.sympify(temp_rhs, locals=locals_map_clairaut))
            y_solved = sympy.solve(temp_eq, y)
            if len(y_solved) == 1 and y_solved[0].is_Add:
                if y_solved[0].coeff(x, 1) == p:
                    f_p = y_solved[0] - x * p
                    if not f_p.has(x) and not f_p.has(y):
                        # Detected Clairaut
                        return _solve_clairaut_step_by_step(temp_eq, y, x, p)
        except Exception:
            pass  # Not Clairaut

        # Standard parsing for other types
        lhs_str, rhs_str = normalized_expr.split('=')
        locals_map = {
            'y': y, 'x': x, 'Derivative': sympy.Derivative,
            'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
            'exp': sympy.exp, 'log': sympy.log
        }
        lhs_sympy = sympy.sympify(lhs_str, locals=locals_map)
        rhs_sympy = sympy.sympify(rhs_str, locals=locals_map)
        equation = sympy.Eq(lhs_sympy, rhs_sympy)
        rearranged = (equation.lhs - equation.rhs).expand()

        # --- CLASSIFICATION CHAIN ---

        # NEW: Check for Linear with Constant Coefficients (handles homogeneous & simple non-homogeneous)
        if _is_linear_constant_coeff(equation, y):
            # This solver handles both homogeneous and simple non-homogeneous forms using undetermined coefficients
            return _solve_linear_constant_coeff_step_by_step(equation, y, x)

        # Check for first-order linear
        if _is_linear(equation, y, x):
            return _solve_linear_step_by_step(equation, y, x)

        # Check for Bernoulli (first-order y' + P y = Q y^n)
        try:
            poly_y = sympy.poly(rearranged, y)
            deg = poly_y.degree()
        except Exception:
            deg = 0

        if deg > 1 and rearranged.has(dydx):
            n = deg
            p_x = poly_y.coeff_monomial(y**1)
            q_x = -poly_y.coeff_monomial(y**n)
            if (sympy.simplify(rearranged - (dydx + p_x * y - q_x * y**n)) == 0 and
                    not p_x.has(y) and not q_x.has(y)):
                return _solve_bernoulli_step_by_step(equation, y, x, n)

        # Check for Homogeneous (first-order)
        f_xy_sol = sympy.solve(equation, dydx)
        if f_xy_sol:
            f_xy = f_xy_sol[0]
            t = sympy.Symbol('t', positive=True)
            if sympy.simplify(f_xy.subs({x: t * x, y: t * y}) - f_xy) == 0:
                return _solve_homogeneous_step_by_step(sympy.Eq(dydx, f_xy), y, x)

        # Check for Exact
        M = rearranged.subs(dydx, 0)
        N = rearranged.coeff(dydx, 1)
        if M != 0 and N != 0 and sympy.simplify(sympy.diff(M, y) - sympy.diff(N, x)) == 0:
            return _solve_exact_step_by_step(M, N, y, x)

        # Fallback: general solver
        solution = sympy.dsolve(equation, y)
        return {
            "solution_summary": sympy.latex(solution),
            "steps": [
                {"rule_name": "Initial Equation", "result": sympy.latex(equation),
                 "explanation": "The problem to be solved."},
                {"rule_name": "General Solution", "result": sympy.latex(solution),
                 "explanation": "The general solution was found using a direct computational method."}
            ]
        }

    except Exception as e:
        return {"error": f"Failed to process equation: {str(e)}"}
