# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import sympy

def solve_differential_equation(expression_str: str):
    """
    Solves a simple, first-order ordinary differential equation string.
    
    Args:
        expression_str: A string representing the equation, e.g., "dy/dx = x**2".

    Returns:
        A dictionary containing the solution summary and steps,
        or an error dictionary if solving fails.
    """
    try:
        # Define the necessary mathematical symbols
        x = sympy.Symbol('x')
        y = sympy.Function('y')(x)
        
        # Prepare the equation parts from the input string
        # IMPORTANT: SymPy's sympify can be a security risk in a production
        # environment if not used with care. For this project, it's okay.
        lhs_str, rhs_str = expression_str.replace(" ", "").split('=')
        
        # Convert the string parts into SymPy expressions
        # We need to replace "dy/dx" with how SymPy understands derivatives: y.diff(x)
        lhs_sympy = sympy.sympify(lhs_str.replace("dy/dx", "y.diff(x)"), locals={'y': y, 'x': x})
        rhs_sympy = sympy.sympify(rhs_str, locals={'y': y, 'x': x})
        
        # Create the equation object
        equation = sympy.Eq(lhs_sympy, rhs_sympy)
        
        # Solve the differential equation
        solution = sympy.dsolve(equation, y)
        
        # Format the solution into the structure our Android app expects
        # NOTE: Generating intermediate steps is a very complex problem.
        # For this server, we will provide the initial and final steps.
        
        # Convert SymPy expressions back to LaTeX strings for the app
        solution_latex = sympy.latex(solution)
        problem_latex = sympy.latex(equation)
        
        result = {
            "solution_summary": solution_latex,
            "steps": [
                {
                    "rule_name": "Initial Equation",
                    "result": problem_latex,
                    "explanation": "This is the ordinary differential equation to be solved."
                },
                {
                    "rule_name": "General Solution",
                    "result": solution_latex,
                    "explanation": "The general solution, including the constant of integration, C1."
                }
            ]
        }
        return result

    except Exception as e:
        # If anything goes wrong (e.g., parsing error), return an error message
        return {
            "error": f"Failed to solve equation: {str(e)}"
        }