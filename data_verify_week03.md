```python
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
```


```python
def verify_file_end(file_path):
    """
    Loads the master data file and displays the last 25 rows, regardless of 
    the total row count, to verify what data is actually present.
    
    Args:
        file_path (str): The path to the master CSV file.
    """
    try:
        # Load the complete master dataframe
        df = pd.read_csv(file_path)
        
        total_rows = len(df)
        logging.info(f"Verification: Total data points found in '{file_path}': {total_rows}")
        
        # Determine the number of rows to display at the end
        rows_to_display = min(25, total_rows)
        
        # Select the last 'rows_to_display' rows
        last_rows = df.tail(rows_to_display)
            
        print(f"\n--- LAST {rows_to_display} ROWS OF bbo_master_w03.csv ---")
        print(f"Total Rows Found: {total_rows}")
        print("---------------------------------------------------------")
        print(last_rows.to_string())
        print("---------------------------------------------------------")
        
        if total_rows < 104:
            logging.warning("The file is incomplete. It should have 104 rows (80 initial + 24 new queries).")
            print("\n**ACTION REQUIRED:** The file is incomplete. You need to re-run the notebook that generated the 24 new queries and appended them to the 80 existing points to reach 104 total rows.")
        else:
            logging.info("File appears to contain the full 104 rows as expected.")
            print("\nFile appears to contain the full 104 rows as expected.")
            print(last_rows.tail(24).to_string())
            

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found.")
        print(f"Error: The file '{file_path}' was not found. Please ensure it exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")


```


```python
# --- Execution ---
if __name__ == '__main__':
    # Using the file you provided
    DATA_FILE = 'bbo_master_w03.csv'
    verify_file_end(DATA_FILE)
```
