```python
import pandas as pd
import numpy as np
import logging
```


```python
# --- Configuration (Matches your original script) ---
DATA_FILE = 'bbo_master_w03.csv'
FUNCTION_CONFIG = {
    1: {'D': 2, 'cols': ['X1', 'X2']},
    2: {'D': 2, 'cols': ['X1', 'X2']},
    3: {'D': 3, 'cols': ['X1', 'X2', 'X3']},
    4: {'D': 4, 'cols': ['X1', 'X2', 'X3', 'X4']},
    5: {'D': 4, 'cols': ['X1', 'X2', 'X3', 'X4']},
    6: {'D': 5, 'cols': ['X1', 'X2', 'X3', 'X4', 'X5']},
    7: {'D': 6, 'cols': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']},
    8: {'D': 8, 'cols': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']},
}
N_FUNCTIONS = 8
```


```python
def check_data_counts(df: pd.DataFrame):
    """
    Checks and prints the number of complete, non-null data points available
    for training each function based on its required dimensionality.
    """
    print(f"\n--- Data Point Training Count Check from {DATA_FILE} ---")
    print("Function ID | Required D | Training Points Found")
    print("-------------------------------------------------")
    
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Data file {DATA_FILE} not found. Cannot perform check.")
        return

    for func_id in range(1, N_FUNCTIONS + 1):
        D = FUNCTION_CONFIG[func_id]['D']
        feature_cols = FUNCTION_CONFIG[func_id]['cols']
        
        # Filter by Function ID
        func_df = df[df['Function ID'] == func_id].copy()
        
        # Count rows where all necessary X columns are NOT null
        complete_rows = func_df.dropna(subset=feature_cols)
        N_data = len(complete_rows)
        
        print(f"{func_id:11} | {D:10} | {N_data:23}")
        
    print("-------------------------------------------------\n")

if __name__ == '__main__':
    # Set up logging to be quiet for the diagnostic script
    logging.basicConfig(level=logging.ERROR)
    check_data_counts(pd.DataFrame()) # Pass an empty DF, the function will load the file

```

    
    --- Data Point Training Count Check from bbo_master_w03.csv ---
    Function ID | Required D | Training Points Found
    -------------------------------------------------
              1 |          2 |                      12
              2 |          2 |                      12
              3 |          3 |                      12
              4 |          4 |                      12
              5 |          4 |                      12
              6 |          5 |                      12
              7 |          6 |                      12
              8 |          8 |                      12
    -------------------------------------------------
    



```python

```


```python

```
