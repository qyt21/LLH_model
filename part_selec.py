import pandas as pd
import numpy as np
import random

def preserve_random_part(input_file, output_file, min_preserve, max_preserve):
    df = pd.read_csv(input_file, header=None)

    def mask_row(row):
        total_points = len(row)  
    
        preserve_count = random.randint(min_preserve, min(max_preserve, int(0.7 * total_points)))  
        
        start_index = random.randint(0, total_points - preserve_count)  
        mask = np.full(total_points, np.nan)  # as NaN
        mask[start_index:start_index + preserve_count] = row[start_index:start_index + preserve_count]
        return pd.Series(mask)  

    masked_data = df.apply(mask_row, axis=1)

    masked_data.to_csv(output_file, index=False, header=False)


input_csv = "input.csv"   
output_csv = "masked_data.csv" 
min_preserve = 10         # minimum to preserve
max_preserve = 30         # maximum to preserve

preserve_random_part(input_csv, output_csv, min_preserve, max_preserve)

print("output.csv created.")
