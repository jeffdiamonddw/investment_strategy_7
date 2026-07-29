import os

import pandas as pd

from utils import s3_list_files



run_name = "2d_test_fold_1"

# pop_files = ["sim_results/{}/populations/{}".format(run_name, filename) for filename in os.listdir("sim_results/{}/populations".format(run_name))]
# df_pops = pd.DataFrame()
# num_done = 0
# for pop_file in pop_files:
#     print("{}/{}".format(num_done, len(pop_files)))
#     df_add = pd.read_parquet(pop_file)
#     df_add['generation'] = int(pop_file.split('.')[0].split('/')[-1].split('_')[1])
#     df_pops = pd.concat([df_pops, df_add])
#     num_done += 1
# df_pops.to_parquet('sim_results/{}/pops.parquet'.format(run_name))


df_median = pd.read_parquet("sim_results/{}/median_objectives".format(run_name))
df_median.to_parquet('sim_results/{}/median_objectives.parquet'.format(run_name))



