from lib.config import *
import numpy as np
import pandas as pd
import os
import warnings
from lib.shapelet import *
from lib.util import *
from tqdm import tqdm


def get_all_model_predictions_week(day_number,type,exclude_models=None):
    if day_number % 7 != 3:
        raise Exception(f"Day Number {day_number} is not valid date")
    if type == 'case':
        path = state_case_dir
    elif type == 'death':
        path = state_death_dir
    models = [model for model in os.listdir(path) if os.path.isdir(f"{path}/{model}")]
    rslt_df_list = []
    for model in models:
        if exclude_models is not None and model in exclude_models:
            continue
        curr_path = f'{path}/{model}/{model}_{day_number}.csv'
        if not os.path.exists(curr_path):
            continue
        curr_df = pd.read_csv(curr_path)
        # if shapelet:
        #     curr_df.iloc[:,1:5] = curr_df.apply(lambda row:shapelet_representation(row.iloc[1:5].astype(np.float64)
        #                                                                ,get_threshold(row.iloc[0])),axis=1,result_type='expand')
        curr_df['model'] = model
        if 'Unnamed: 0' in curr_df.columns:
            curr_df = curr_df.drop('Unnamed: 0',axis=1)
        rslt_df_list.append(curr_df.dropna())
    if len(rslt_df_list) == 0:
        return None
    else:
        return pd.concat(rslt_df_list)

def get_all_model_predictions_week_dict(day_number,shapelet=False):
    """ Function used to generate dist_mat

    :param day_number:
    :param shapelet:
    :return:
    """
    rslt = {}
    if day_number % 7 != 3:
        raise Exception(f"Day Number {day_number} is not valid date")
    path = state_case_dir
    models = [model for model in os.listdir(path) if os.path.isdir(f"{path}/{model}")]
    for model in models:
        curr_path = f'{path}/{model}/{model}_{day_number}.csv'
        if not os.path.exists(curr_path):
            continue
        curr_df = pd.read_csv(curr_path)
        if shapelet:
            curr_df.iloc[:,1:5] = curr_df.apply(lambda row:shapelet_representation(row.iloc[1:5].astype(np.float64)
                                                                       ,Thresholds.loc[row.iloc[0]][0]),axis=1,result_type='expand')
        curr_df['model'] = model
        for row in curr_df.iterrows():
            row = row[1]
            rslt[(day_number,row['model'],row['State'])] = np.array(row.iloc[1:5]).astype(np.float64)
    return rslt
def get_all_predictions_in_period(day_number, type, lookback_period, exclude_models=None):
    weeks_of_interest = [int(day_number - i * 7) for i in range(lookback_period)]
    df_list = {}
    for day in weeks_of_interest:
        df = get_all_model_predictions_week(day,type,exclude_models=exclude_models)
        if df is None:
            warnings.warn(f"Day {day} does not have any valid prediction!")
        df_list[day] = df
    return df_list
def get_all_predictions_in_period_dict(day_number, lookback_period, shapelet=False):
    weeks_of_interest = [int(day_number - i * 7) for i in range(lookback_period)]
    rslt_dict_all = {}
    for day in weeks_of_interest:
        rslt_dict = get_all_model_predictions_week_dict(day,shapelet=shapelet)
        if rslt_dict is None:
            warnings.warn(f"Day {day} does not have any valid prediction!")
        rslt_dict_all.update(rslt_dict)
    return rslt_dict_all

def get_4weeks_ahead_ground_truth(start_day,type,shapelet=False):
    dates_of_interest = [str(int(start_day + 7 * i)) for i in range(4)]
    return get_ground_truth_array(dates_of_interest,type,shapelet=shapelet)

def get_ground_truth_array(day_arr,type,shapelet=False):
    if type == 'case':
        infile = state_case_actual_file
    elif type == 'death':
        infile = state_death_actual_file
    else:
        raise BaseException(f"Invalid Type {type}")
    day_arr = [str(elem) for elem in day_arr]
    rslt = pd.read_csv(infile,index_col=[0],usecols=['State']+day_arr)
    if shapelet:
        rslt = pd.DataFrame({rslt.index[i]:shapelet_representation(rslt.iloc[i]) for i in range(len(rslt.index))}).T
        rslt.columns = day_arr
    return rslt.sort_index()

def get_all_model_predictions():
    path = state_case_dir
    rslt = {}
    models = os.listdir(path)
    for model in models:
        df_list_model = [pd.read_csv(f"{path}/{model}/{file}",index_col=[0]) for file
                         in os.listdir(f"{path}/{model}")]
        df_weeks_ahead = {}
        empty_flag = True
        for i in range(4):
            to_concat = [df.iloc[:, i+1] for df in df_list_model if df.shape[1] >= i+2]
            if len(to_concat) == 0:
                continue
            else:
                empty_flag = False
                df_weeks_ahead[i] = pd.concat(to_concat, axis=1)
                df_weeks_ahead[i].columns = df_weeks_ahead[i].columns.astype(np.int64)
        if not empty_flag:
            rslt[model] = df_weeks_ahead
    return rslt
