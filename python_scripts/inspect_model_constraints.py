import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn
import pandas as pd

def get_constraint_matrix(model):
    """
    Extracts linear constraints from a Pyomo model into an A*x + constant <= 0 format.
    Returns a pandas DataFrame where each row is a constraint and columns are variables + constant.
    """
    # 1. Collect all active variable objects and their names
    all_vars = sorted(
        list(model.component_data_objects(pyo.Var, active=True)), 
        key=lambda x: x.name
    )
    var_names = [v.name for v in all_vars]
    matrix_data = []

    # 2. Iterate through all active constraints in the model
    for c_comp in model.component_objects(pyo.Constraint, active=True, descend_into=True):
        for index in c_comp:
            c = c_comp[index]
            
            # Generate linear representation
            repn = generate_standard_repn(c.body)
            
            # Check for non-linear terms
            if repn.nonlinear_vars:
                print(f"Skipping nonlinear constraint: {c.name}")
                continue

            # Initialize row with zeros using variable names (strings)
            row = {name: 0.0 for name in var_names}
            
            # Fill coefficients using the variable object's name
            for v, coeff in zip(repn.linear_vars, repn.linear_coefs):
                row[v.name] = coeff
            
            # 3. Format as A*x + constant <= 0
            # If constraint is body <= upper: body - upper <= 0
            # If constraint is body >= lower: lower - body <= 0 (Multiply by -1)
            # If constraint is equality: body - value <= 0
            if c.has_ub():
                row['constant'] = repn.constant - c.upper
            elif c.has_lb():
                # Multiply coefficients and constant by -1 to flip to <= 0
                for v in row:
                    row[v] = -row[v]
                row['constant'] = c.lower - repn.constant
            else: # Equality constraint
                row['constant'] = repn.constant - c.lower
            
            matrix_data.append(row)
            
    # Return as DataFrame, filling missing variable coefficients with 0
    return pd.DataFrame(matrix_data).fillna(0)



import pyomo.environ as pyo


def get_infeasibility_model(_model):
    """
    Replaces existing constraints: Ax + constant <= 0 
    with: Ax + constant <= q, and minimizes q.
    """
    model = _model.clone()
    
    # 1. Add slack variable q
    model.q = pyo.Var(domain=pyo.NonNegativeReals)
    
    # 2. Deactivate original objective and set new objective
    if hasattr(model, 'OBJ'):
        model.OBJ.deactivate()
    model.MIN_Q = pyo.Objective(expr=model.q, sense=pyo.minimize)
    
    # 3. Rebuild constraints
    # We must iterate over existing constraints and re-define them
    constraint_names = [c.name for c in model.component_objects(pyo.Constraint, active=True)]
    
    for name in constraint_names:
        c_comp = getattr(model, name)
        
        # We need to store the rules or expressions to rebuild them
        # Note: This assumes constraints were added via rules
        # If they were added via expressions, this logic needs to be adapted
        for index in c_comp:
            # Reconstruct the body - this is the most reliable way 
            # to inject the 'q' slack
            expr = c_comp[index].body
            
            # Deactivate the old constraint
            c_comp[index].deactivate()
            
            # Create a new constraint: body <= q + upper_bound
            # Assuming your original constraints were: body <= 0
            new_expr = expr <= model.q
            setattr(model, f"{name}_{index}_slack", pyo.Constraint(expr=new_expr))
            
    return model