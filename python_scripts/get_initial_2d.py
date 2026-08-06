import numpy as np
import pandas as pd

random_initial_file = 's3://jdinvestment/2d_initial/populations/gen_0.parquet'
initial_file_0d = 's3://jdinvestment/0d_no_kits/populations/gen_149.parquet'

df_random = pd.read_parquet(random_initial_file)
df_0d = pd.read_parquet(initial_file_0d)

num_add_rows = df_random.shape[0] - df_0d.shape[0]
add_rows = np.random.choice(range(df_0d.shape[0]), size=num_add_rows, replace=True)
df_0d = pd.concat([df_0d, df_0d.iloc[add_rows]])

df_random.to_parquet('s3://jdinvestment/2d_initial/populations/gen_0_random.parquet')
df_random.loc[:, df_0d.columns] = df_0d.values
df_random.to_parquet('s3://jdinvestment/2d_initial/populations/gen_0.parquet')

zzz=1