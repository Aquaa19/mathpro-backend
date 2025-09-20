# FILE: solver.py
# LOCATION: mathpro-backend/solver.py

import sympy
import re

def normalize_expression(expr: str) -> str:
    """
    Cleans up a user-input string to be compatible with SymPy.
    - Replaces superscript characters (², ³) with caret notation (^2, ^3).
    - Replaces caret notation (x^2) with Python's power operator (x**2).
    """
    # Dictionary to map superscript characters to standard form
    superscript_map = {
        '²': '^2', '³': '^3', '¹': '^1', '⁰': '^0',
        '⁴': '^4', '⁵': '^5', '⁶': '^6', '⁷': '^7',
        '⁸': '^8', '⁹': '^9'
    }

    # First pass: replace all superscripts
    for sup, std in superscript_map.items():
        expr = expr.replace(sup, std)
    
    # Second pass: use a regular expression to replace x^2 with x**2
    # This is safer than a simple replace('^', '**')
    # It looks for a character/number followed by ^ and a number
    expr = re.sub(r'(\w+)\^(\d+)', r'\1**\2', expr)
    
    return expr

def solve_differential_equation(expression_str: str):
    """
    Solves a simple, first-order ordinary differential equation string.
    """
    try:
        # 1. Normalize the user's input first!
        normalized_expr = normalize_expression(expression_str)
        
        # 2. Define the mathematical symbols
        x = sympy.Symbol('x')
        y = sympy.Function('y')(x)
        
        # 3. Prepare the equation parts from the normalized string
        lhs_str, rhs_str = normalized_expr.replace(" ", "").split('=')
        
        lhs_sympy = sympy.sympify(lhs_str.replace("dy/dx", "y.diff(x)"), locals={'y': y, 'x': x})
        rhs_sympy = sympy.sympify(rhs_str, locals={'y': y, 'x': x})
        
        equation = sympy.Eq(lhs_sympy, rhs_sympy)
        solution = sympy.dsolve(equation, y)
        
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
        return {
            "error": f"Failed to solve equation: {str(e)}"
        }