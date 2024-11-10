import pandas as pd
import glob
import os

folder_path = 'DATA'

def calculate_diversification_score(row):
    risk_adjusted_return = (row['sharpe_ratio'] + row['treynor_ratio']) / 2
    
    capture_ratio = row['up_capture_ratio'] / row['down_capture_ratio'] if row['down_capture_ratio'] != 0 else 0
    
    alpha_beta_ratio = row['alpha'] / row['beta'] if row['beta'] != 0 else 0
    
    diversification_score = risk_adjusted_return + capture_ratio + alpha_beta_ratio
    
    return diversification_score

for file_path in glob.glob(os.path.join(folder_path, 'chunk_*.csv')):
    df = pd.read_csv(file_path)
    
    required_columns = ['sharpe_ratio', 'treynor_ratio', 'up_capture_ratio', 'down_capture_ratio', 'alpha', 'beta']
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {file_path}, required columns missing.")
        continue
    
    df['diversification_score'] = df.apply(calculate_diversification_score, axis=1)
    
    new_file_path = file_path.replace('.csv', '_with_score.csv')
    df.to_csv(new_file_path, index=False)
    print(f"Processed and saved {new_file_path}")
