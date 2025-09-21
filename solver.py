# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import re
import sympy

# ===== REPLACEMENT: normalize_expression =====
def normalize_expression(expr: str) -> str:
    """
    Clean and normalize a user input string into something SymPy-friendly.
    - maps common unicode superscripts and sqrt,
    - replaces prime and Leibniz derivative notations with SymPy Derivative(...) text,
    - inserts a limited set of implicit multiplications,
    - converts ^ to **,
    - returns a string ready for sympy.sympify with appropriate locals.
    """
    import re

    if not isinstance(expr, str):
        raise TypeError("expr must be a string")

    expr = expr.strip()

    # Map unicode to ascii/plain forms
    char_map = {
        '²': '^2', '³': '^3', '¹': '^1', '⁰': '^0', '⁴': '^4', '⁵': '^5',
        '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '’': "'", '”': "''", '‘': "'",
        '√': 'sqrt'
    }
    for k, v in char_map.items():
        expr = expr.replace(k, v)

    # Normalize whitespace
    expr = re.sub(r'\s+', ' ', expr).strip()

    # --- Replace Leibniz style derivatives like dy/dx or d^2 y / dx^2 ---
    deriv_map = {}
    counter = 0

    def new_placeholder():
        nonlocal counter
        ph = f"__DERIV_{counter}__"
        counter += 1
        return ph

    pattern_leibniz = re.compile(r'd\^?\s*(\d+)?\s*y\s*/\s*dx\^?\s*(\d+)?', flags=re.IGNORECASE)

    def _replace_leibniz(m):
        n1 = m.group(1)
        n2 = m.group(2)
        if n1 and n1.isdigit():
            n = int(n1)
        elif n2 and n2.isdigit():
            n = int(n2)
        else:
            n = 1
        ph = new_placeholder()
        deriv_map[ph] = f"Derivative(y(x),(x,{n}))" if n != 1 else "Derivative(y(x),x)"
        return ph

    expr = pattern_leibniz.sub(_replace_leibniz, expr)

    # --- Replace prime notation y', y'' etc. ---
    pattern_primes = re.compile(r"y('{1,})")

    def _replace_primes(m):
        quotes = m.group(1)
        n = len(quotes)
        ph = new_placeholder()
        deriv_map[ph] = f"Derivative(y(x),(x,{n}))" if n != 1 else "Derivative(y(x),x)"
        return ph

    expr = pattern_primes.sub(_replace_primes, expr)

    # --- Insert simple implicit multiplications ---
    # Keep this conservative to avoid breaking function names
    func_names = r'(?:sin|cos|tan|log|exp|sqrt|Derivative|__DERIV_\d+__)'

    # between number and identifier or '('
    expr = re.sub(
        rf'(?P<num>(?<![A-Za-z_])\d+(?:\.\d+)?)(?P<next>(?:{func_names}|[A-Za-z_]\w*|\())',
        r'\g<num>*\g<next>', expr)

    # between closing bracket/identifier and opening bracket/identifier
    expr = re.sub(
        rf'(?P<left>(?:[A-Za-z0-9_)\]]|__DERIV_\d+__))(?P<right>(?:\(|{func_names}|[A-Za-z_]))',
        r'\g<left>*\g<right>', expr)

    # derivative placeholders atomization
    expr = re.sub(r'(__DERIV_\d+__)(?=[A-Za-z_(])', r'\1*', expr)
    expr = re.sub(r'(?<=[A-Za-z0-9_)])(__DERIV_\d+__)', r'*\1', expr)

    # caret to python power
    expr = expr.replace('^', '**')

    # replace placeholders with derivative text
    for ph, sym_text in deriv_map.items():
        expr = expr.replace(ph, sym_text)

    # final cleanup: remove accidental spaces
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

# ===== REPLACEMENT: _solve_linear_constant_coeff_step_by_step =====
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
    m = sympy.Symbol('m')
    aux_eq_poly = lhs.subs({y.diff(x, i): m**i for i in range(10, 0, -1)}).subs(y, 1)
    aux_eq = sympy.Eq(aux_eq_poly, 0)
    steps.append({"rule_name": "Step 1: Find the Complementary Function (CF)", "result": "First, solve the homogeneous equation: " + sympy.latex(sympy.Eq(lhs, 0)), "explanation": "The general solution is y = y_c + y_p."})
    steps.append({"rule_name": "Form the Auxiliary Equation", "result": sympy.latex(aux_eq), "explanation": "Substitute y = e^(mx) into the homogeneous equation."})
    
    roots = sympy.roots(aux_eq_poly, m)
    steps.append({"rule_name": "Find the Roots", "result": f"m = {sympy.latex(list(roots.keys()))}", "explanation": "Solve the polynomial for its roots."})

    cf_terms = []
    root_index = 0
    for root, multiplicity in roots.items():
        if sympy.im(root) == 0: # Real Roots
            for i in range(multiplicity):
                cf_terms.append(C_symbols[root_index] * (x**i) * sympy.exp(root * x))
                root_index += 1
        else: # Complex Roots
            alpha, beta = sympy.re(root), sympy.im(root)
            if beta > 0:
                for i in range(multiplicity):
                    term = sympy.exp(alpha*x) * (C_symbols[root_index] * sympy.cos(beta*x) + C_symbols[root_index+1] * sympy.sin(beta*x)) * (x**i)
                    cf_terms.append(term)
                    root_index += 2
    
    complementary_function = sum(cf_terms)
    steps.append({"rule_name": "Construct the Complementary Function", "result": f"y_c = {sympy.latex(complementary_function)}", "explanation": "Construct the CF from the roots."})

    # --- Part 2: Find the Particular Integral (PI) ---
    if rhs == 0:
        final_solution = sympy.Eq(y, complementary_function)
        steps.append({"rule_name": "General Solution (Homogeneous)", "result": sympy.latex(final_solution), "explanation": "For a homogeneous equation, the general solution is the complementary function."})
        return {"solution_summary": sympy.latex(final_solution), "steps": steps}
    
    steps.append({"rule_name": "Step 2: Find the Particular Integral (PI)", "result": "Find a particular solution y_p for the non-homogeneous equation.", "explanation": "Use the Method of Undetermined Coefficients."})

    # Guess the form of the PI using SymPy's built-in helper
    pi_guess = sympy.solvers.ode.undetermined_coeffs.undetermined_coeffs_set(rhs, x).pop()
    steps.append({"rule_name": "Guess the Form of y_p", "result": f"y_p = {sympy.latex(pi_guess)}", "explanation": f"Based on f(x) = {sympy.latex(rhs)}, we guess the form of the particular solution."})

    # Check for duplication with CF and modify guess if needed
    temp_pi_guess = pi_guess
    while any(sympy.solvers.ode.undetermined_coeffs.check_assumptions(term, temp_pi_guess) for term in cf_terms):
        temp_pi_guess *= x
    
    if temp_pi_guess != pi_guess:
        pi_guess = temp_pi_guess
        steps.append({"rule_name": "Modify Guess for Duplication", "result": f"y_p = {sympy.latex(pi_guess)}", "explanation": "The initial guess duplicates a term in the CF. Multiply the guess by x until it is independent."})

    # Substitute the guess into the equation and solve for coefficients
    eq_for_coeffs = sympy.Eq(lhs.subs(y, pi_guess).doit(), rhs)
    undetermined_coeffs = pi_guess.atoms(sympy.Symbol) - {x}
    solved_coeffs = sympy.solve(eq_for_coeffs, *undetermined_coeffs)
        
    particular_integral = pi_guess.subs(solved_coeffs)
    steps.append({"rule_name": "Solve for Coefficients", "result": f"y_p = {sympy.latex(particular_integral)}", "explanation": f"Substitute y_p into the original equation to solve for the coefficients."})

    # --- Part 3: Combine for General Solution ---
    final_solution = sympy.Eq(y, complementary_function + particular_integral)
    steps.append({"rule_name": "Step 3: Construct General Solution (y = y_c + y_p)", "result": sympy.latex(final_solution), "explanation": "The full solution is the sum of the complementary function and the particular integral."})
    
    return {"solution_summary": sympy.latex(final_solution), "steps": steps}



# --- REPLACED: helper detection function ----------------------------------

# ===== REPLACEMENT: _is_linear_constant_coeff =====
# --- REPLACE this helper function ---

def _is_linear_constant_coeff(eq, y):
    """Checks if an equation is a linear ODE with constant coefficients."""
    try:
        # We only care about the coefficients of y and its derivatives on the LHS
        rearranged = eq.lhs - eq.rhs
        lhs = sympy.collect(rearranged, y.atoms(sympy.Derivative).union({y}))

        # Check that all coefficients of y and its derivatives are constants
        # and that there are no terms with y mixed with derivatives
        poly = sympy.poly(lhs, y.atoms(sympy.Derivative).union({y}))
        return all(c.is_constant(x=False) for c in poly.coeffs())
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
        if _is_linear_constant_coeff(equation, y, x):
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
