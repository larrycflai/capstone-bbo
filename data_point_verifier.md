```python
import numpy as np
import os

BASE_DATA_PATH = 'initial_data' 
NUM_FUNCTIONS = 8
```


```python
print("--- Initial Data File Verification ---")
print("Function ID | Inputs Shape (Rows, Dims) | Initial Data Points")
print("-" * 55)

for fn_id in range(1, NUM_FUNCTIONS + 1):
    func_folder = os.path.join(BASE_DATA_PATH, f'function_{fn_id}')
    inputs_path = os.path.join(func_folder, 'initial_inputs.npy')

    try:
        # Load the input data file
        X_inputs = np.load(inputs_path)
        
        # The first element of the shape tuple is the number of rows
        num_rows = X_inputs.shape[0]
        
        # The shape of the array
        shape_str = f"({X_inputs.shape[0]}, {X_inputs.shape[1]})"
        
        # Print the findings
        print(f"F{fn_id:<2}          | {shape_str:<23} | {num_rows}")

    except FileNotFoundError:
        print(f"F{fn_id:<2}          | *** FILE NOT FOUND ***")
    except Exception as e:
        print(f"F{fn_id:<2}          | Error loading: {e}")

print("-" * 55)
print("Conclusion: The total number of points is the first number in the shape tuple.")
```

    --- Initial Data File Verification ---
    Function ID | Inputs Shape (Rows, Dims) | Initial Data Points
    -------------------------------------------------------
    F1           | (10, 2)                 | 10
    F2           | (10, 2)                 | 10
    F3           | (15, 3)                 | 15
    F4           | (30, 4)                 | 30
    F5           | (20, 4)                 | 20
    F6           | (20, 5)                 | 20
    F7           | (30, 6)                 | 30
    F8           | (40, 8)                 | 40
    -------------------------------------------------------
    Conclusion: The total number of points is the first number in the shape tuple.



```python

```


```python

```
