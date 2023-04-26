import os.path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

debug = False

# Directories for files
raw_dir = 'Data/Raw'
state_case_dir = f'Data/Cleaned/State_Case'
state_death_dir = f'Data/Cleaned/State_Death'
state_convert_file = "Data/state_convert.csv"
state_pop_file = "Data/state_population.csv"

# base_dir = 'Data_Sources/Cases/Input_Files/Model_Forecasts_Data/US-COVID'
# county_case_dir = f'{base_dir}/country-case'
# state_death_dir = f'{base_dir}/state-death'
state_case_quantile_dir = "Data/Cleaned/State_Case_Quantile"
state_death_quantile_dir = "Data/Cleaned/State_Death_Quantile"
state_case_actual_file = 'Data/state_case_inc.csv'
state_death_actual_file = 'Data/state_death_inc.csv'
distance_matrix_folder = 'dist_mats'
result_output_dir = 'result/cases'

output_dir_case_quantile = 'result/Quantiles_Normpop/case'
output_dir_death_quantile = 'result/Quantiles_Normpop/death'

# Configuration for shapelet
shapelet_array = np.zeros(shape=(4,4))
shapelet_array[0] = [1,2,3,4] # inc
shapelet_array[1] = [1,2,2,1] # peak
shapelet_array[2] = [0,0,0,0] # flat
shapelet_array[3] = [4,3,2,1] # dec
shapelet_array_names = ['Inc','Peak','Flat','Dec']
Shapelet_length = 4

# Configuration for lookback period
Lookback_Period = 6

# Configuration for shapelets
Threshold_End_Date = '2020-06-01'
threshold_folder = 'threshold'

# Confirguation for cluster
K_cluster = 3

# Configuration for predict models
Model = RandomForestRegressor()

# Clustering Lookback
Cluster_Lookback_Period = 1

# Quantiles defined in forecast hub
# target_quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
#                     0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]
target_quantiles = [0.025,0.1,0.25,0.5,0.75,0.9,0.975]

Thresholds = pd.read_csv(f"{threshold_folder}/threhold_{Threshold_End_Date}.csv",index_col=[0])




