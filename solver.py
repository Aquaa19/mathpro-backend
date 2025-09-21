# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import re
import sympy

# --- Helpers & Normalization -------------------------------------------------

def normalize_expression(expr: str) -> str:
    """
    Clean up user input to make it SymPy-friendly:
    - Normalize Unicode superscripts/primes to caret/Derivative form
    - Handle caret -> ** for powers
    - Add conservative implicit multiplication rules
    - Convert prime/Leibniz notations into Derivative(y(x), (x,n)) textual form
    """
    if not isinstance(expr, str):
        raise TypeError("normalize_expression expects a string")

    # Remove spaces to simplify regex handling (we'll be careful with tokens)
    expr = expr.replace(" ", "")

    # --- Unicode superscripts & quote mapping ---
    char_map = {
        '²': '^2', '³': '^3', '¹': '^1', '⁰': '^0',
        '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7',
        '⁸': '^8', '⁹': '^9', '’': "'", '”': "''", '‘': "'",
    }
    for special, standard in char_map.items():
        expr = expr.replace(special, standard)

    # --- Convert prime notation for y (y', y'', y''') into Derivative(...) ---
    # Handle y(x) or y
    expr = re.sub(r"y\(\s*x\s*\)'''", r"Derivative(y(x),(x,3))", expr)
    expr = re.sub(r"y\(\s*x\s*\)''", r"Derivative(y(x),(x,2))", expr)
    expr = re.sub(r"y\(\s*x\s*\)'", r"Derivative(y(x),x)", expr)

    expr = re.sub(r"y'''", r"Derivative(y(x),(x,3))", expr)
    expr = re.sub(r"y''", r"Derivative(y(x),(x,2))", expr)
    expr = re.sub(r"y'", r"Derivative(y(x),x)", expr)

    # --- Convert Leibniz notation: d^n y / dx^n or dy/dx ---
    # Examples handled:
    #   dy/dx    -> Derivative(y(x), x)
    #   d^2y/dx^2 -> Derivative(y(x),(x,2))
    # Accept optional ^ or not and optional spaces (we removed spaces earlier)
    # Pattern: d^?N y / dx^?M
    def _leibniz_repl(match):
        n = match.group(1)
        m = match.group(2)
        if n is None:
            n_val = 1
        else:
            n_val = int(n)
        if m is None:
            m_val = 1
        else:
            m_val = int(m)
        if n_val != m_val:
            # if mismatch, prefer max(n,m) for safety (user input ambiguity)
            n_val = max(n_val, m_val)
        if n_val == 1:
            return "Derivative(y(x),x)"
        else:
            return f"Derivative(y(x),(x,{n_val}))"

    expr = re.sub(r"d\^?(\d+)?y/dx\^?(\d+)?", _leibniz_repl, expr)

    # Also match forms like d2y/dx2 (no ^)
    expr = re.sub(r"d(\d+)y/dx(\d+)", _leibniz_repl, expr)

    # dy/dx (simple)
    expr = expr.replace("dy/dx", "Derivative(y(x),x)")

    # --- Implicit multiplication (conservative rules) ---
    # 1) Digit followed by letter or '('  ->  2x  -> 2*x,  2(x+1) -> 2*(x+1)
    expr = re.sub(r'(\d)([A-Za-z(])', r'\1*\2', expr)

    # 2) ')' followed by '(' or letter/digit -> ')( ' -> ')*(' or ')x' -> ')*x'
    expr = re.sub(r'(\))([A-Za-z0-9(])', r'\1*\2', expr)

    # 3) letter (single-letter variable) followed by digit -> x2 -> x*2
    expr = re.sub(r'([A-Za-z])(\d)', r'\1*\2', expr)

    # Note: we avoid aggressive splitting of multi-letter function names (sin, cos, log, etc.)
    # so we don't insert '*' inside "sin(x)" etc.

    # --- Caret -> Python power operator ---
    # Replace caret with ** (safe because we removed spaces and caret is used for exponents)
    expr = expr.replace("^", "**")

    return expr


# --- Step-by-step solver for first-order linear ODEs -------------------------

def _solve_linear_step_by_step(eq, y_func, x_sym):
    """
    Generate a detailed step-by-step solution for first-order linear ODEs of form:
        y'(x) + P(x) * y(x) = Q(x)
    Inputs:
        eq        : a SymPy Eq object representing the ODE
        y_func    : sympy.Function('y') call y_func(x_sym)
        x_sym     : sympy.Symbol('x')
    Returns:
        dict with solution_summary and steps list
    """
    # Coerce forms: left should be dydx + P(x)*y
    # We take eq.lhs - eq.rhs = 0 and inspect coefficients
    p_x = None
    q_x = None

    # Extract P(x) and Q(x) based on the standard rearrangement
    rearranged = (eq.lhs - eq.rhs).expand()
    dydx = sympy.Derivative(y_func(x_sym), x_sym)
    # Collect by derivative and y(x)
    collected = sympy.collect(rearranged, [dydx, y_func(x_sym)])

    # Coefficient of derivative
    coeff_dydx = collected.coeff(dydx, 1)
    if coeff_dydx != 1:
        # Try dividing through by coefficient (if it is nonzero)
        if coeff_dydx != 0:
            collected = sympy.simplify(collected / coeff_dydx)
        # re-evaluate coefficient
        coeff_dydx = collected.coeff(dydx, 1)

    p_x = collected.coeff(y_func(x_sym), 1)
    # Q(x) is everything else after extracting dydx and p_x*y
    q_x_expr = -(collected - dydx - p_x * y_func(x_sym))
    # ensure p_x and q_x don't contain y(x)
    if p_x.has(y_func(x_sym)) or q_x_expr.has(y_func(x_sym)):
        raise ValueError("P(x) or Q(x) still contain y(x) — not linear in standard form")

    steps = []
    steps.append({
        "rule_name": "Identify P(x) and Q(x)",
        "result": f"P(x) = {sympy.latex(p_x)}, Q(x) = {sympy.latex(q_x_expr)}",
        "explanation": "Recognized first-order linear ODE of form y' + P(x) y = Q(x)."
    })

    # Integrating factor
    integral_p = sympy.integrate(p_x, x_sym)
    integrating_factor = sympy.exp(integral_p)
    steps.append({
        "rule_name": "Calculate Integrating Factor (I.F.)",
        "result": f"I(x) = e^{{\\int P(x) dx}} = {sympy.latex(integrating_factor)}",
        "explanation": "Compute e^(∫P(x) dx)."
    })

    # Multiply and integrate
    multiplied_lhs = sympy.latex(integrating_factor * (sympy.Derivative(y_func(x_sym), x_sym) + p_x * y_func(x_sym)))
    multiplied_rhs = sympy.latex(integrating_factor * q_x_expr)
    steps.append({
        "rule_name": "Multiply Equation by I.F.",
        "result": f"{multiplied_lhs} = {multiplied_rhs}",
        "explanation": "Multiplying the ODE by the integrating factor makes the left side the derivative of (I(x)*y)."
    })

    integral_of_rhs = sympy.integrate(integrating_factor * q_x_expr, x_sym)
    steps.append({
        "rule_name": "Integrate Both Sides",
        "result": f"{sympy.latex(integrating_factor)} y = \\int {multiplied_rhs} dx = {sympy.latex(integral_of_rhs)} + C",
        "explanation": "Integrate both sides with respect to x."
    })

    C = sympy.Symbol('C')
    final_solution = sympy.Eq(y_func(x_sym), (integral_of_rhs + C) / integrating_factor)
    steps.append({
        "rule_name": "Final Solution",
        "result": sympy.latex(final_solution),
        "explanation": "Solve for y(x) to get the general solution."
    })

    return {"solution_summary": sympy.latex(final_solution), "steps": steps}


# --- Main solver & classifier ------------------------------------------------

def solve_differential_equation(expression_str: str):
    """
    Classify and solve a differential equation given as a string.
    Returns a dict: either {"solution_summary": ..., "steps": [...]} or {"error": "..."}
    """
    try:
        # Symbols / function objects
        x = sympy.Symbol('x')
        y_func = sympy.Function('y')  # y as a function object
        yx = y_func(x)

        # Normalize input
        normalized = normalize_expression(expression_str)

        # Split into LHS and RHS (must have exactly one '=')
        if normalized.count('=') != 1:
            return {"error": "Input must contain exactly one '=' separating LHS and RHS."}
        lhs_str, rhs_str = normalized.split('=')

        # Prepare locals mapping for sympify
        # Provide common math functions to sympy parser
        safe_locals = {
            'x': x,
            'y': y_func,           # allow 'y(x)' syntax
            'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
            'log': sympy.log, 'exp': sympy.exp, 'sqrt': sympy.sqrt,
            'Derivative': sympy.Derivative,
        }

        # Convert to sympy expressions
        lhs_sym = sympy.sympify(lhs_str, locals=safe_locals)
        rhs_sym = sympy.sympify(rhs_str, locals=safe_locals)
        equation = sympy.Eq(lhs_sym, rhs_sym)

        # Try to detect first-order linear ODE: y'(x) + P(x)*y(x) = Q(x)
        try:
            rearranged = (equation.lhs - equation.rhs).expand()
            dydx = sympy.Derivative(yx, x)
            collected = sympy.collect(rearranged, [dydx, yx])
            coeff_dydx = collected.coeff(dydx, 1)
            # Normalize if derivative coefficient not 1 but non-zero
            if coeff_dydx != 1 and coeff_dydx != 0:
                collected = sympy.simplify(collected / coeff_dydx)

            # Re-evaluate after possible normalization
            coeff_dydx = collected.coeff(dydx, 1)
            px_candidate = collected.coeff(yx, 1)

            # Check px_candidate and remainder for y(x)
            qx_candidate = -(collected - dydx - px_candidate * yx)

            if coeff_dydx == 1 and (not px_candidate.has(yx)) and (not qx_candidate.has(yx)):
                # It's a first-order linear ODE
                linear_eq = sympy.Eq(dydx + px_candidate * yx, qx_candidate)
                return _solve_linear_step_by_step(linear_eq, y_func, x)
        except Exception:
            # classification attempt failed — fall back to general solver below
            pass

        # Fallback: use sympy.dsolve for general ODE solving
        solution = sympy.dsolve(equation, y_func(x))
        return {
            "solution_summary": sympy.latex(solution),
            "steps": [
                {"rule_name": "Initial Equation", "result": sympy.latex(equation),
                 "explanation": "Parsed equation."},
                {"rule_name": "General Solution", "result": sympy.latex(solution),
                 "explanation": "General solution found using SymPy's solver."}
            ]
        }

    except Exception as e:
        return {"error": f"Failed to process equation: {str(e)}"}
