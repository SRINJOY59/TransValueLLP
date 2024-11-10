import pandas as pd
import os

def split_csv(file_path, output_dir, chunk_size=20 * 1024 * 1024):  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_size = 0
    chunk_number = 1

    for chunk in pd.read_csv(file_path, chunksize=100000):  
        output_file = os.path.join(output_dir, f"chunk_{chunk_number}.csv")
        chunk.to_csv(output_file, index=False)
        chunk_size_mb = os.path.getsize(output_file)

        total_size += chunk_size_mb
        if total_size >= chunk_size:
            chunk_number += 1
            total_size = 0

    print(f"CSV file split into {chunk_number} chunks.")


file_path = "ALL_SCHEMES_DATA.csv"

output_dir = "DATA"

split_csv(file_path, output_dir, chunk_size=20 * 1024 * 1024)  
