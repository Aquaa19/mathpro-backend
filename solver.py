# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import sympy

# --- Step-by-Step Solvers ---

def _solve_linear_step_by_step(eq, y, x):
    """Generates detailed steps for first-order linear equations."""
    
    # Form: dy/dx + P(x)*y = Q(x)
    # 1. Identify P(x) and Q(x)
    p_x = eq.lhs.coeff(y, 1)
    q_x = eq.rhs
    
    steps = [{
        "rule_name": "Identify P(x) and Q(x)",
        "result": f"P(x) = {sympy.latex(p_x)}, Q(x) = {sympy.latex(q_x)}",
        "explanation": "The equation is a first-order linear differential equation of the form dy/dx + P(x)y = Q(x)."
    }]
    
    # 2. Calculate the integrating factor
    integral_p_x = sympy.integrate(p_x, x)
    integrating_factor = sympy.exp(integral_p_x)
    
    steps.append({
        "rule_name": "Calculate Integrating Factor (I.F.)",
        "result": f"I(x) = e^{{\\int P(x) dx}} = {sympy.latex(integrating_factor)}",
        "explanation": "The integrating factor is calculated as e raised to the power of the integral of P(x)."
    })
    
    # 3. Multiply the entire equation by the integrating factor
    multiplied_lhs = sympy.latex(integrating_factor * (y.diff(x) + p_x * y))
    multiplied_rhs = sympy.latex(integrating_factor * q_x)
    
    steps.append({
        "rule_name": "Multiply Equation by I.F.",
        "result": f"{multiplied_lhs} = {multiplied_rhs}",
        "explanation": "Multiplying by the integrating factor makes the left side the derivative of (I.F. * y)."
    })
    
    # 4. Integrate both sides
    integral_of_rhs = sympy.integrate(integrating_factor * q_x, x)
    
    steps.append({
        "rule_name": "Integrate Both Sides",
        "result": f"{sympy.latex(integrating_factor)} y = \\int {multiplied_rhs} dx = {sympy.latex(integral_of_rhs)} + C",
        "explanation": "The equation is integrated with respect to x to solve for y."
    })
    
    # 5. Form the final solution
    final_solution = sympy.Eq(y, (integral_of_rhs + sympy.Symbol('C')) / integrating_factor)
    
    steps.append({
        "rule_name": "Final Solution",
        "result": sympy.latex(final_solution),
        "explanation": "Isolating y gives the general solution to the differential equation."
    })
    
    return {
        "solution_summary": sympy.latex(final_solution),
        "steps": steps
    }

# --- Main Solver and Classifier ---

def solve_differential_equation(expression_str: str):
    """
    Classifies and solves a differential equation, providing steps.
    """
    try:
        # Standard setup
        x = sympy.Symbol('x')
        y = sympy.Function('y')(x)
        dydx = y.diff(x)

        # Normalize user input for exponents, etc.
        normalized_expr = expression_str.replace("^", "**")
        lhs_str, rhs_str = normalized_expr.replace(" ", "").split('=')
        
        # Create the SymPy equation object
        lhs_sympy = sympy.sympify(lhs_str.replace("dy/dx", "dydx"), locals={'y': y, 'x': x, 'dydx': dydx})
        rhs_sympy = sympy.sympify(rhs_str, locals={'y': y, 'x': x, 'dydx': dydx})
        
        equation = sympy.Eq(lhs_sympy, rhs_sympy)
        
        # --- CLASSIFICATION LOGIC ---
        
        # Attempt to classify as First-Order Linear: dy/dx + P(x)y = Q(x)
        # We rearrange the equation to the form: dydx + ... = 0
        rearranged_eq = equation.lhs - equation.rhs
        
        # Use 'collect' to group terms by y and its derivative
        collected_eq = sympy.collect(rearranged_eq.expand(), [y, dydx])
        
        # Check if the coefficient of dydx is 1 and the coefficient of y is independent of y
        if collected_eq.coeff(dydx, 1) == 1 and not collected_eq.coeff(y, 1).has(y):
            # It's a linear equation!
            p_x = collected_eq.coeff(y, 1)
            q_x = -(collected_eq - p_x * y - dydx) # The rest of the equation is Q(x)
            
            if not p_x.has(y) and not q_x.has(y): # Ensure P(x) and Q(x) are functions of x only
                print("Classifier: Detected First-Order Linear Equation.")
                linear_eq = sympy.Eq(dydx + p_x * y, q_x)
                return _solve_linear_step_by_step(linear_eq, y, x)

        # --- FALLBACK ---
        # If no specific type is matched, use the general solver
        print("Classifier: No specific type matched. Using general solver.")
        solution = sympy.dsolve(equation, y)
        return {
            "solution_summary": sympy.latex(solution),
            "steps": [
                {"rule_name": "Initial Equation", "result": sympy.latex(equation), "explanation": "The problem to be solved."},
                {"rule_name": "General Solution", "result": sympy.latex(solution), "explanation": "The general solution was found using a direct computational method."}
            ]
        }

    except Exception as e:
        return {"error": f"Failed to process equation: {str(e)}"}