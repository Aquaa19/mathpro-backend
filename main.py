# FILE: main.py
# LOCATION: mathpro-backend/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Import our actual solver function from the other file
from solver import solve_differential_equation

# --- Pydantic Models ---
# These models define the structure of the JSON data for our API.
# FastAPI uses them to validate incoming requests and format outgoing responses.

class Step(BaseModel):
    rule_name: str
    result: str
    explanation: Optional[str] = None

class SolveRequest(BaseModel):
    expression: str
    # The 'options' map from our Android app can be added here if needed
    # options: Optional[dict] = None

class SolveResponse(BaseModel):
    solution_summary: str
    steps: List[Step]

# Create the main FastAPI application instance
app = FastAPI(
    title="MathPRO API",
    description="An API for solving differential equations.",
    version="1.0.0"
)

# --- API Endpoint ---

@app.post("/solve", response_model=SolveResponse)
async def solve_endpoint(request: SolveRequest):
    """
    Receives a mathematical expression, solves it, and returns the result.
    """
    print(f"Received expression to solve: {request.expression}")
    
    # Call our solver function from solver.py
    result = solve_differential_equation(request.expression)
    
    # Check if the solver returned an error
    if "error" in result:
        # If there's an error, raise an HTTPException, which FastAPI
        # will convert into a 400 Bad Request error response.
        raise HTTPException(status_code=400, detail=result["error"])
    
    # If successful, return the result. FastAPI will ensure it matches
    # the SolveResponse model and convert it to JSON.
    return result