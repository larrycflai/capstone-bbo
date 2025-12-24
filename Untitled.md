```python
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import logging
import warnings
import json
import os
from typing import List, Tuple, Dict, Any

# Import necessary components from scikit-learn for GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# --- Configuration ---
DATA_FILE_PATH = 'bbo_master_w06.csv' # Adjust if your master file name has changed
ADD_DATA_DIR = 'add_data'
OUTPUT_FILE_PATH = os.path.join(ADD_DATA_DIR, 'week07_clean_inputs.json')

# Define the true dimensionality (D) for each of the 8 functions
FUNCTION_DIMENSIONS = {
    1: 2, 2: 2, 3: 3, 4: 4,
    5: 4, 6: 5, 7: 6, 8: 8
}

# Set up logging for cleaner output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Suppress GPR numerical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def expected_improvement(X, gp, f_best):
    """Acquisition function: Expected Improvement (EI)"""
    # X must be reshaped if it's a single point (1D array to 2D array)
    X = np.atleast_2d(X)
    
    # Calculate mean and standard deviation
    mu, sigma = gp.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1) # Ensure sigma is 2D
    
    # Calculate difference and z-score
    diff = mu - f_best
    with np.errstate(divide='ignore'): # Ignore division by zero warnings
        Z = diff / sigma
        
    # Calculate Expected Improvement (EI)
    # The EI formula is: (mu - f_best) * CDF(Z) + sigma * PDF(Z)
    ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Optimization aims for minimisation, so we return the negative EI
    return -ei.flatten()


def process_function(function_id: int, df_all: pd.DataFrame) -> Tuple[List[float], float]:
    """
    Optimises the next query point for a single function using GPR and EI.
    
    This function includes the crucial fix for NaN values by selecting only 
    the relevant X columns (X1 to XD).
    """
    D = FUNCTION_DIMENSIONS.get(function_id)
    if D is None:
        logging.error(f"Missing dimension for Function F{function_id}. Skipping.")
        return [], 0.0

    logging.info(f"--- Processing Function F{function_id} ({D}D) ---")
    
    # 1. Prepare Data
    df_func = df_all[df_all['Function ID'] == function_id].copy()
    
    # X_columns are the columns X1, X2, ..., X8
    X_cols = [f'X{i}' for i in range(1, 9)]
    X_train_raw = df_func[X_cols].values
    Y_train = df_func['Y'].values

    # CRUCIAL FIX: Filter out the NaN columns based on the true dimension (D)
    # This selects X1, ..., XD and avoids feeding NaNs to the GPR.
    X_train_clean = X_train_raw[:, :D]
    
    # Check for NaNs just in case the data frame was modified elsewhere
    if np.isnan(X_train_clean).any():
         # This should now only trigger if there are NaNs within the relevant columns
         logging.error(f"Error processing Function F{function_id}: Input X still contains NaN within columns X1-X{D}. Returning random query.")
         # Fallback to a random query if cleaning somehow failed
         return np.random.uniform(0.0, 1.0, D).tolist(), 0.0

    logging.info(f"Loaded {len(X_train_clean)} points for F{function_id}. Fixed Dimension D={D} enforced.")

    # Scale the Y values for better GPR performance
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
    
    # The maximum (best) value in the scaled space
    f_best_scaled = np.max(Y_scaled)
    logging.info(f"Best known MAXIMUM (Scaled Y) for F{function_id}: {f_best_scaled:.4f}")

    # 2. Train GPR Model
    # Using Matern kernel (v=2.5) with a flexible length-scale and constant amplitude
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0]*D, length_scale_bounds=(1e-2, 1e3), nu=2.5) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=1e-6, 
        n_restarts_optimizer=15, 
        normalize_y=False # Already scaled manually
    )

    try:
        gp.fit(X_train_clean, Y_scaled)
        # logging.info(f"GPR Model Trained. Log-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.15f}")
    except ValueError as e:
        # If GPR fit fails, return a random query
        logging.error(f"Error fitting GPR for F{function_id}: {e}. Returning random query.")
        return np.random.uniform(0.0, 1.0, D).tolist(), 0.0

    # 3. Optimise Acquisition Function (Expected Improvement)
    
    # We use a multi-start optimisation approach to find the global minimum of -EI
    n_restarts = 10
    best_query = None
    max_ei = -np.inf

    # Define the objective function wrapper for minimizer
    objective = lambda x: expected_improvement(x, gp, f_best_scaled)
    # Define bounds for the optimisation (0.0 to 1.0 for all D dimensions)
    bounds = [(0.0, 1.0)] * D
    
    for i in range(n_restarts):
        # Start search from a random point
        x0 = np.random.uniform(0.0, 1.0, D)
        
        # Use L-BFGS-B minimisation algorithm
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds
        )
        
        # If a better point (lower negative EI, thus higher positive EI) is found
        if res.success and -res.fun[0] > max_ei:
            max_ei = -res.fun[0]
            best_query = res.x
            
    logging.info(f"Optimal EI found: {max_ei:.4f}")

    # 4. Format Output
    if best_query is None:
        # Fallback if optimisation fails completely
        logging.warning(f"Optimisation failed for F{function_id}. Returning random query.")
        return np.random.uniform(0.0, 1.0, D).tolist(), 0.0

    # Clip values to ensure they are strictly within [0, 1] for submission
    final_query = np.clip(best_query, 0.0, 1.0).tolist()
    logging.info(f"F{function_id} Proposed Query (D={D}): " + ' | '.join([f'{x:.6f}' for x in final_query]))

    return final_query, max_ei


def run_optimisation():
    """Main function to run the optimisation loop for all 8 functions."""
    
    if not os.path.exists(ADD_DATA_DIR):
        os.makedirs(ADD_DATA_DIR)
        
    try:
        df_master = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        logging.error(f"Could not find the master data file at: {DATA_FILE_PATH}. Please ensure it exists.")
        return

    # Ensure all inputs are treated as numbers, errors will be handled by the check inside process_function
    df_master.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Iterate over all 8 functions
    all_queries = []
    
    for func_id in range(1, 9):
        new_query, ei_value = process_function(func_id, df_master)
        
        # Pad the query with 0.0s up to D=8 for submission format
        # IMPORTANT: The API only uses the first D dimensions, but the JSON must be padded to D=8 for all entries.
        padded_query = new_query + [0.0] * (8 - len(new_query))
        all_queries.append(padded_query)

    # 5. Save Results to JSON
    logging.info("\n--------------------------------------------------")
    logging.info("SUCCESS: Generated 8 FINAL queries for Week 7 (Rounds 0-16).")
    logging.info(f"File saved to: '{OUTPUT_FILE_PATH}'. This is the file to submit.")
    
    with open(OUTPUT_FILE_PATH, 'w') as f:
        # Format the JSON data to match the required submission format (array of arrays)
        json.dump(all_queries, f, indent=2, allow_nan=False)
        
    logging.info("--------------------------------------------------")


if __name__ == '__main__':
    run_optimisation()
```

    INFO: --- Processing Function F1 (2D) ---
    ERROR: Error processing Function F1: Input X still contains NaN within columns X1-X2. Returning random query.
    INFO: --- Processing Function F2 (2D) ---
    ERROR: Error processing Function F2: Input X still contains NaN within columns X1-X2. Returning random query.
    INFO: --- Processing Function F3 (3D) ---
    ERROR: Error processing Function F3: Input X still contains NaN within columns X1-X3. Returning random query.
    INFO: --- Processing Function F4 (4D) ---
    ERROR: Error processing Function F4: Input X still contains NaN within columns X1-X4. Returning random query.
    INFO: --- Processing Function F5 (4D) ---
    ERROR: Error processing Function F5: Input X still contains NaN within columns X1-X4. Returning random query.
    INFO: --- Processing Function F6 (5D) ---
    ERROR: Error processing Function F6: Input X still contains NaN within columns X1-X5. Returning random query.
    INFO: --- Processing Function F7 (6D) ---
    ERROR: Error processing Function F7: Input X still contains NaN within columns X1-X6. Returning random query.
    INFO: --- Processing Function F8 (8D) ---
    ERROR: Error processing Function F8: Input X still contains NaN within columns X1-X8. Returning random query.
    INFO: 
    --------------------------------------------------
    INFO: SUCCESS: Generated 8 FINAL queries for Week 7 (Rounds 0-16).
    INFO: File saved to: 'add_data/week07_clean_inputs.json'. This is the file to submit.
    INFO: --------------------------------------------------

