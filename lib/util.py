import numpy as np
import os
import pandas as pd
import warnings
from .config import *
from tqdm import tqdm

def get_all_states():
    return np.array(['Washington', 'Illinois', 'California', 'Arizona', 'Massachusetts',
       'Wisconsin', 'Texas', 'Nebraska', 'Utah', 'Oregon', 'Florida',
       'New York', 'Rhode Island', 'Georgia', 'New Hampshire',
       'North Carolina', 'New Jersey', 'Colorado', 'Maryland', 'Nevada',
       'Tennessee', 'Hawaii', 'Indiana', 'Kentucky', 'Minnesota',
       'Oklahoma', 'Pennsylvania', 'South Carolina', 'Kansas', 'Missouri', 'Vermont',
       'Virginia', 'Connecticut', 'Iowa', 'Louisiana', 'Ohio', 'Michigan',
       'South Dakota', 'Arkansas', 'Delaware', 'Mississippi',
       'New Mexico', 'North Dakota', 'Wyoming', 'Alaska', 'Maine',
       'Alabama', 'Idaho', 'Montana','West Virginia'], dtype=object)

def normalize(arr):
   if (arr == 0).all():
      return arr
   return (arr - arr.mean()) / (np.sqrt(len(arr)) * arr.std())

def folder(path):
   if not os.path.exists(path):
      os.makedirs(path)
   return path

def valid_forecasthub_day(day):
   return (day > 0) and (day % 7 == 3)

def convert_date_to_forecasthub_num(day_str):
   return (pd.to_datetime(day_str) - pd.to_datetime('2020-1-23')).days + 1

def convert_forecasthub_truth(in_file=None, out_file=None):
   df = pd.read_csv(in_file)
   names = pd.read_csv('Data/state_convert.csv')['location_name'].unique()
   df = df[df['location_name'].isin(names)][['date', 'location_name', 'value']]
   df = df.pivot(index='date', columns='location_name', values='value')
   df = df.rolling(7).sum() # Aggregate a week of incident data
   df.index = df.index.map(convert_date_to_forecasthub_num)
   df.T.to_csv(out_file)

def convert_forecasthub_to_quantile_format(inpath, outpath, target):
   models = os.listdir(inpath)
   conversion_map = pd.read_csv('Data/state_convert.csv', index_col=['location'])['location_name'].to_dict()
   for model in tqdm(models):
      # if model != "USC-SI_kJalpha":
      #    continue
      files = [file for file in os.listdir(f"{inpath}/{model}") if file[-4:] == '.csv']
      for file in files:
         forecast_date_str = '-'.join(file.split('-')[:3])
         # if forecast_date_str != '2021-08-01':
         #    continue
         forecast_date = (pd.to_datetime(forecast_date_str) - pd.to_datetime('2020-1-23')).days + 1
         forecast_date = int(np.floor(forecast_date / 7) * 7) + 3
         curr_df = pd.read_csv(f"{inpath}/{model}/{file}")
         curr_df = curr_df[(curr_df['type'] == 'quantile')].dropna()
         curr_df = curr_df[curr_df['location'].astype(str).str.isnumeric()]
         curr_df['location'] = curr_df['location'].astype(np.int64).map(conversion_map)
         curr_df = curr_df.dropna()
         curr_df = curr_df[curr_df['target'].str.contains(f"wk ahead inc {target}")] # day for hosp
         curr_df['target'] = curr_df['target'].str.split().str.get(0).astype(np.int64)
         curr_df = curr_df[curr_df['target'].isin([1, 2, 3, 4])]
         curr_df['target_end_date'] = (pd.to_datetime(curr_df['target_end_date']) - pd.to_datetime(
            '2020-1-23')).dt.days + 1
         curr_df = curr_df.drop(['forecast_date', 'type','target'], axis=1)
         curr_df['target_end_date'] = (np.ceil(curr_df['target_end_date'] / 7) * 7 + 3).astype(np.int32)
         if len(curr_df) == 0:
            continue
         else:
            path = folder(f"{outpath}/{model}")
            curr_df.to_csv(f"{path}/{model}_{forecast_date}.csv")


def convert_forecasthub_to_median_format(inpath, outpath, type):
   paths = os.listdir(inpath)
   for model_name in tqdm(paths):
      path = f"{inpath}/{model_name}"
      files = [file for file in os.listdir(path) if file[-4:] == '.csv']
      conversion_map = pd.read_csv(state_convert_file, index_col=['location'],low_memory=False)['location_name'].to_dict()
      for file in files:
         forecast_date_str = '-'.join(file.split('-')[:3])
         # forecast_date = (pd.to_datetime(forecast_date_str) - pd.to_datetime('2020-1-23')).days + 1
         # forecast_date = int(np.floor(forecast_date / 7) * 7) + 3
         curr_df = pd.read_csv(f"{path}/{file}")
         curr_df = curr_df[(curr_df['type'] == 'quantile') & (curr_df['quantile'] == 0.5)]
         curr_df = curr_df[curr_df['location'].astype(str).str.isnumeric()]
         curr_df['location'] = curr_df['location'].astype(np.int64).map(conversion_map)
         curr_df = curr_df.dropna()
         curr_df = curr_df[curr_df['target'].str.contains(f'wk ahead inc {type.lower()}')]
         curr_df['target'] = curr_df['target'].str.split().str.get(0).astype(np.int64)
         curr_df = curr_df[curr_df['target'].isin([1, 2, 3, 4])]
         curr_df['target_end_date'] = (pd.to_datetime(curr_df['target_end_date']) - pd.to_datetime(
            '2020-1-23')).dt.days + 1
         curr_df = curr_df[['location', 'target_end_date', 'value']]
         curr_df['value'] = np.round(curr_df['value']).astype(int)
         if len(curr_df) == 0:
            continue
         to_concat = []
         for i in curr_df.groupby(['location']):
            curr_loc = i[0][0]
            df_i = list(i[1].set_index('target_end_date').T.iloc[1])
            to_concat.append([curr_loc] + df_i)
         rslt = pd.DataFrame(to_concat)
         targets = sorted(curr_df['target_end_date'].unique())[:4]
         for i in range(len(targets)):
            if targets[i] % 7 != 3:
               warnings.warn("Model end date not coherent with forecast hub format!")
               targets[i] = int(np.ceil(i / 7) * 7 + 3)
         rslt.columns = ['State'] + targets
         folder(f"{outpath}/{model_name}")
         rslt.set_index("State").to_csv(f"{outpath}/{model_name}/{model_name}_{targets[0]}.csv")


def clean_formatted_model(model_folder, out_dir):
   model_name = model_folder.split('/')[-1]
   for file in os.listdir(model_folder):
      curr_file_path = f"{model_folder}/{file}"
      if not os.path.isfile(curr_file_path):
         warnings.warn(f"Non-file entity {file} found in folder {model_folder}!")
         continue
      pred_day = int(file.split('_')[-1][:-4])
      pred_day_adjusted = int(np.floor(pred_day / 7) * 7) + 3
      first_target_date = pred_day_adjusted + 7
      target_date_range = [str(int(first_target_date + i * 7)) for i in range(4)]
      curr_df = pd.read_csv(f"{curr_file_path}").set_index('State')
      curr_df = curr_df[curr_df.columns[curr_df.columns.isin(target_date_range)]]
      if curr_df.shape[1] != 4:
         warnings.warn(f"File {file} does not have 4 weeks ahead of predictions! Skipping")
         continue
      curr_out_dir = folder(f"{out_dir}/{model_name}")
      curr_df.to_csv(f"{curr_out_dir}/{model_name}_{first_target_date}.csv")

def clean_formatted_all_models(in_folder=None, out_folder=None):
   if in_folder is None:
      in_folder = state_case_dir
   if out_folder is None:
      out_folder = "State_Case"
   model_dirs = [curr_folder for curr_folder in os.listdir(in_folder) if os.path.isdir(f"{in_folder}/{curr_folder}")]
   for model_dir in tqdm(model_dirs):
      clean_formatted_model(f"{in_folder}/{model_dir}", out_folder)

def get_days_from_dict(df_dict):
   return sorted(list(set([elem[0] for elem in list(df_dict.keys())])))

def get_models_from_dict_by_day(df_dict,target_day_arr):
   return sorted(list(set([elem[1] for elem in list(df_dict.keys()) if elem[0] in target_day_arr])))

def get_threshold(state,type):
   if type == 'death':
      return 0
   return Thresholds.loc[state][0]
def compute_threshold(end_date, case_death='case'):
   df = pd.read_csv(state_case_actual_file, index_col=0)
   df_weekly_inc = df.rolling(8).agg(lambda x: x.iloc[-1] - x.iloc[0])
   return df_weekly_inc.diff().abs().loc[:end_date].rolling(3).mean().max()

def convert_quantile_to_result_format(inmodel,outdir,type='case'):
   if type == 'case':
      path = f"{state_case_quantile_dir}/{inmodel}"
   elif type == 'death':
      path = f"{state_death_quantile_dir}/{inmodel}"

   for file in tqdm(os.listdir(path)):
      day = int(file[:-4].split('_')[-1])
      curr_df = pd.read_csv(f'{path}/{file}')
      first_target_day = sorted(curr_df['target_end_date'].unique())[0]
      for target_day in curr_df['target_end_date'].unique():
         curr_df_target = curr_df[curr_df['target_end_date'] == target_day][['location','value','quantile']]
         curr_df_target = curr_df_target.pivot(index='location', columns='quantile', values='value')
         outpath = folder(f"{outdir}/{inmodel}/{inmodel}_{first_target_day}")
         curr_df_target.to_csv(f"{outpath}/{inmodel}_{target_day}.csv")




