import pyarrow.parquet as pq
import pandas as pd

def load_sample(file_path, n_rows=100000):
    pq_file = pq.ParquetFile(file_path)
    batch = next(pq_file.iter_batches(batch_size=n_rows))
    df = batch.to_pandas()
    return df

if __name__ == "__main__":
    path = r"D:\IIT-Gandhinagar_Project\dataset_10M.parquet"
    
    df = load_sample(path)
    
    df.to_csv(r"D:\IIT-Gandhinagar_Project\sample_100k.csv", index=False)
    
    print("✅ Sample CSV created!")