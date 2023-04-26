import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
from lib import *
from quantile_forest import RandomForestQuantileRegressor
import warnings
import concurrent.futures
import functools

class InvalidStateException(Exception):
    "Raised when the input value is less than 18"
    pass

def get_model_state_helper(df, model, state, show_warnings=False):
    ts = df[(df['model'] == model) & (df['State'] == state)]
    if len(ts) == 0:
        return None
    if show_warnings:
        if len(ts) != 1:
            warnings.warn(f"State {state} for model {model} contains duplicate results!")
    return ts.iloc[0,1:5].values.astype(np.float64).flatten()


def preprocess(df_list, cluster_mapping, add_shapelet=False):
    def _preprocess(weeks):
        train_set = []
        for week in weeks:
            curr_df = df_list[week]
            if len(curr_df) == 0:
                warnings.warn(f"No data for week {week}")
                continue
            curr_df['bin'] = curr_df['model'].map(cluster_mapping)
            curr_df = curr_df.set_index('State').sort_values('bin')
            grouped_median = curr_df.groupby(['bin','State']).median().reset_index()
            grouped_median_expanded = [x.set_index('State').iloc[:,1:] for _, x in grouped_median.groupby(['bin'])]
            ground_truth_wk = int(curr_df.columns[0])
            ground_truth = get_4weeks_ahead_ground_truth(ground_truth_wk, shapelet=False)
            if add_shapelet:
                grouped_median_shpaes = grouped_median.copy()
                grouped_median_shpaes.iloc[:, -4:] = grouped_median_shpaes.iloc[:, -4:].apply(
                    lambda row: shapelet_representation(np.array(row), get_threshold(row.iloc[0])),
                    axis=1, result_type='broadcast')
                grouped_median_shpaes = grouped_median_shpaes.groupby(['State']).median().iloc[:, -4:]
                curr_train_set = pd.concat(grouped_median_expanded,axis=1).dropna()\
                    .join(grouped_median_shpaes,rsuffix='_shape').join(ground_truth,rsuffix='_true')
            else:
                curr_train_set = pd.concat(grouped_median_expanded,axis=1).dropna().join(ground_truth,rsuffix='_true')

            curr_train_set.columns = range(len(curr_train_set.columns))
            train_set.append(curr_train_set)
        return pd.concat(train_set).dropna()

    weeks = sorted(list(df_list.keys()))
    test_weeks = [weeks.pop(-1)]
    train_weeks = weeks
    train_set = _preprocess(train_weeks)
    test_set = _preprocess(test_weeks)
    X_train = train_set.iloc[:,:-4]
    y_train = train_set.iloc[:,-4:]
    X_test = test_set.iloc[:,:-4]
    y_test = test_set.iloc[:,-4:]
    # if normalize_y and not shapelet:
    #     y_train = y_train.apply(lambda row:normalize(row), axis=1)
    #     y_test = y_test.apply(lambda row:normalize(row), axis=1)
    return X_train,X_test,y_train,y_test


def compute_dist_day(target_day):
    if not valid_forecasthub_day(target_day):
        raise Exception(f"Day {target_day} is not a valid forecast hub target date")
    print(f"Process for day {target_day} started!")
    df_dict = get_all_predictions_in_period_dict(target_day, Lookback_Period, shapelet=True)
    all_states = get_all_states()
    models = get_models_from_dict_by_day(df_dict,[target_day - i*7 for i in range(Lookback_Period)])
    if len(models) == 0:
        warnings.warn(f"No data to generate distance matrix for {target_day}")
        return
    day_targets = get_days_from_dict(df_dict)
    num_models = len(models)
    num_states = len(all_states)
    dist = np.full((Lookback_Period, num_states, num_models, num_models), np.nan)
    for i in range(len(day_targets)):
        for j in range(len(all_states)):
            for k, l in combinations(range(len(models)), 2):
                key1 = (day_targets[i], models[k], all_states[j])
                key2 = (day_targets[i], models[l], all_states[j])
                if (key1 in df_dict) and (key2 in df_dict):
                    ts_model1 = df_dict[(day_targets[i], models[k], all_states[j])]
                    ts_model2 = df_dict[(day_targets[i], models[l], all_states[j])]
                    curr_dist = cosine(ts_model1,ts_model2)
                    dist[i, j, k, l] = round(curr_dist,3)
                    dist[i, j, l, k] = round(curr_dist,3)

    dist_matrix = pd.DataFrame(np.nanmean(dist,axis=(1,0)),columns=models,index=models)
    np.fill_diagonal(dist_matrix.values, 0)
    out_dir = folder(f"{distance_matrix_folder}/dist_mat_{Lookback_Period}")
    dist_matrix.to_csv(f"{out_dir}/{target_day}.csv")
    print(f"Computation for day {target_day} Completed!")

def compute_dist(start_day,end_day):
    if start_day % 7 != 3:
        start_day = int(np.ceil(start_day/7) * 7) + 3

    with concurrent.futures.ProcessPoolExecutor() as executor:
        day_range = range(start_day, end_day, 7)
        executor.map(compute_dist_day,day_range)

def check_mapping_validity(df_list,cluster_mapping,target_day_range,k_cluster):
    for target_day in target_day_range:
        df_mapping = pd.Series(cluster_mapping)
        for cluster_bin in range(k_cluster):
            curr_df_mapping = df_mapping[df_mapping == cluster_bin]
            valid = False
            for model_name in curr_df_mapping.index:
                avaliable_models = df_list[target_day]['model'].unique()
                # If a week does not have any result at all, the preprocessing step skips it so it's taken care of
                if len(avaliable_models) == 0 or model_name in df_list[target_day]['model'].unique():
                    valid=True
                    continue
            if not valid:
                return False
    return True
def extract_weeks_ahead(week_ahead,X_train,X_test,y_train,y_test):
    X_train_curr = pd.concat([X_train.iloc[:,-Shapelet_length:],
                              X_train.iloc[:,:-Shapelet_length].iloc[:,week_ahead::4]],axis=1)
    y_train_curr = y_train.iloc[:,week_ahead]
    X_test_curr = pd.concat([X_test.iloc[:,-Shapelet_length:],
                              X_test.iloc[:,:-Shapelet_length].iloc[:,week_ahead::4]],axis=1)
    y_test_curr = y_test.iloc[:,week_ahead]
    return X_train_curr,X_test_curr,y_train_curr,y_test_curr
def run_sim(target_day):
    if not valid_forecasthub_day(target_day):
        raise Exception("Not a valid COVID-Hub forecast day!")
    print(f"Thread run_sim for day {target_day} begins")
    df_list = get_all_predictions_in_period(target_day, Lookback_Period)
    models = df_list[target_day]['model'].unique()
    rslt_weeks = df_list[target_day].columns[1:5]
    dist_file = f"{distance_matrix_folder}/{target_day}_{Lookback_Period}.csv"
    if not os.path.exists(dist_file):
        warnings.warn(f"No file found for distance matrix {dist_file}! Please run compute_dist first.")
        return
    dist_matrix = pd.read_csv(dist_file,index_col=[0])
    valid_cluster = False
    curr_k_cluster = K_cluster
    while not valid_cluster:
        cluster = make_cluster(K_cluster, dist_matrix.values, method='constrained')
        cluster_mapping = {model: bin for model, bin in zip(models, cluster)}
        if not check_mapping_validity(df_list, cluster_mapping, list(df_list.keys()), curr_k_cluster):
            warnings.warn(f"Invalid k={curr_k_cluster}. Trying again with k={curr_k_cluster - 1}")
            curr_k_cluster -= 1
            print(curr_k_cluster)
            if curr_k_cluster == 0:
                raise "Clustering failed, curr_k_cluster = 0"
        else:
            valid_cluster = True
    X_train,X_test,y_train,y_test = preprocess(df_list, cluster_mapping, add_shapelet=True)
    states = X_test.index
    rslt = pd.DataFrame(np.zeros(shape=(len(states),len(rslt_weeks))))
    rslt.index = states
    rslt.columns = rslt_weeks
    model_name = type(Model).__name__
    for i in range(4):
        X_train_curr, X_test_curr, y_train_curr, y_test_curr = extract_weeks_ahead(i, X_train, X_test, y_train, y_test)
        Model.fit(X_train_curr.values,y_train_curr.values)
        pred_current = Model.predict(X_test_curr)
        rslt.iloc[:,i] = pred_current
    out_folder = folder(f"{result_output_dir}/{model_name}_{Lookback_Period}_{K_cluster}")
    rslt.to_csv(f"{out_folder}/{model_name}_{Lookback_Period}_{curr_k_cluster}_{target_day}.csv")
    print(f"Thread run_sim for day {target_day} ends")
def run_sim_range(start_day,end_day):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        day_range = range(start_day,end_day,7)
        executor.map(run_sim,day_range)

# def preprocess_without_clustering(df_list,type,add_shapelet=False,normpop=False,median_first=True):
#     def _preprocess(weeks):
#         train_set = []
#         for week in weeks:
#             curr_df = df_list[week]
#             if len(curr_df) == 0:
#                 warnings.warn(f"No data for week {week}")
#                 continue
#             curr_df = curr_df.set_index('State').drop(['model'],axis=1)
#             curr_df_copy = curr_df.copy()
#             ground_truth_wk = int(curr_df.columns[0])
#             ground_truth = get_4weeks_ahead_ground_truth(ground_truth_wk,type,shapelet=False)
#             if normpop:
#                 pop_df = pd.read_csv(state_pop_file).set_index('State')
#                 for i in range(curr_df.shape[0]):
#                     curr_df.iloc[i] = curr_df.iloc[i]/pop_df.loc[curr_df.iloc[i].name].values[0] * 100000
#                 for i in range(ground_truth.shape[0]):
#                     ground_truth.iloc[i] = ground_truth.iloc[i]/pop_df.loc[curr_df.iloc[i].name].values[0] * 100000
#             if add_shapelet:
#                 shape_values = pd.DataFrame(curr_df_copy.apply(
#                     lambda row: shapelet_representation(np.array(row.values), m_0 = get_threshold(row.name,type)),
#                     axis=1).to_list()).values
#                 grouped_median_shapes = pd.DataFrame(columns=curr_df_copy.columns,index=curr_df_copy.index,data=shape_values)
#                 curr_train_set = pd.concat([curr_df,grouped_median_shapes],axis=1)
#                 if median_first:
#                     curr_train_set = curr_train_set.groupby(level=0).median()
#                 curr_train_set = curr_train_set.join(ground_truth,rsuffix='_true')
#             else:
#                 curr_train_set = curr_df.join(ground_truth,rsuffix='_true')
#
#             curr_train_set.columns = range(len(curr_train_set.columns))
#             train_set.append(curr_train_set)
#         return pd.concat(train_set).dropna()
#
#     weeks = sorted(list(df_list.keys()))
#     test_weeks = [weeks.pop(-1)]
#     train_weeks = weeks
#     train_set = _preprocess(train_weeks)
#     test_set = _preprocess(test_weeks)
#     X_train = train_set.iloc[:,:-4]
#     y_train = train_set.iloc[:,-4:]
#     X_test = test_set.iloc[:,:-4]
#     y_test = test_set.iloc[:,-4:]
#     return X_train,X_test,y_train,y_test

def preprocess_without_clustering(df_list, type, add_shapelet=False, normpop=False, median_first=True):
    def _preprocess(weeks):
        train_set = []
        for week in weeks:
            curr_df = df_list[week]
            if len(curr_df) == 0:
                warnings.warn(f"No data for week {week}")
                continue

            curr_df = curr_df.set_index('State').drop(['model'], axis=1)
            ground_truth_wk = int(curr_df.columns[0])
            ground_truth = get_4weeks_ahead_ground_truth(ground_truth_wk, type, shapelet=False)

            if normpop:
                pop_df = pd.read_csv(state_pop_file).set_index('State')
                norm_factors_curr_df = pop_df.loc[curr_df.index].values.flatten()
                norm_factors_ground_truth = pop_df.loc[ground_truth.index].values.flatten()
                curr_df = curr_df.div(norm_factors_curr_df, axis=0) * 100000
                ground_truth = ground_truth.div(norm_factors_ground_truth, axis=0) * 100000

            if add_shapelet:
                shape_values = curr_df.apply(
                    lambda row: shapelet_representation(np.array(row.values), m_0=get_threshold(row.name, type)),
                    axis=1).to_list()
                grouped_median_shapes = pd.DataFrame(columns=curr_df.columns, index=curr_df.index, data=shape_values)
                curr_train_set = pd.concat([curr_df, grouped_median_shapes], axis=1)
            else:
                curr_train_set = curr_df.copy()

            if median_first:
                curr_train_set = curr_train_set.groupby(level=0).median()
                curr_train_set = curr_train_set.join(ground_truth, rsuffix='_true')

            else:
                curr_train_set = curr_df.join(ground_truth, rsuffix='_true')

            curr_train_set.columns = range(len(curr_train_set.columns))
            train_set.append(curr_train_set)

        return pd.concat(train_set).dropna()

    weeks = sorted(list(df_list.keys()))
    test_weeks = [weeks.pop(-1)]
    train_weeks = weeks
    train_set = _preprocess(train_weeks)
    test_set = _preprocess(test_weeks)
    X_train = train_set.iloc[:, :-4]
    y_train = train_set.iloc[:, -4:]
    X_test = test_set.iloc[:, :-4]
    y_test = test_set.iloc[:, -4:]
    return X_train, X_test, y_train, y_test

def generate_quantile_prediction(start_day, end_day, type, add_shapelet=False,normpop=False,median_first=True):
    partial_func = functools.partial(generate_quantile_prediction_day, type=type, add_shapelet=add_shapelet,
                                     normpop=normpop,median_first=median_first)
    day_range = range(start_day, end_day, 7)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(partial_func, day_range)

def generate_quantile_prediction_day(day,type,add_shapelet,normpop=False,tree_depth=None,median_first=True):
    model_name = "RandomForestQuantileRegressor"
    if tree_depth is not None:
        model_name += f'Depth={tree_depth}'
    if normpop:
        model_name += 'Normpop'
    if add_shapelet:
        model_name += '_shapelet'

    print(f"Quantile prediction for day {day} started!")
    target_days = [int(day + 7 * i) for i in range(4)]
    exclude_models = ['COVIDhub-4_week_ensemble',"COVIDhub-baseline","COVIDhub-ensemble","COVIDhub-trained_ensemble",
                      "COVIDhub_CDC-ensemble"]
    df_list = get_all_predictions_in_period(day,type,Lookback_Period, exclude_models=exclude_models)
    X_train, X_test, y_train_all, y_test_all = preprocess_without_clustering(df_list,type,add_shapelet=add_shapelet,normpop=normpop)
    for i in range(4):
        y_train_curr = y_train_all.iloc[:, i]
        qclf_curr = RandomForestQuantileRegressor(max_depth=tree_depth).fit(X_train, y_train_curr)
        y_pred_curr = qclf_curr.predict(X_test, quantiles=target_quantiles)
        curr_rslt = pd.DataFrame(y_pred_curr, index=y_test_all.index, columns=target_quantiles)
        if not median_first:
            curr_rslt = curr_rslt.groupby(level=0).median()
        if normpop:
            pop_df = pd.read_csv(state_pop_file).set_index('State')
            for j in range(curr_rslt.shape[0]):
                curr_rslt.iloc[j] = np.round(curr_rslt.iloc[j] * pop_df.loc[curr_rslt.iloc[j].name].values[0] / 100000)
                curr_out_path = folder(f"result/Quantiles_Normpop/{type}/{model_name}/{model_name}_{day}/")
        else:
            curr_out_path = folder(f"result/Quantiles/{type}/{model_name}/{model_name}_{day}/")

        curr_rslt.to_csv(
            f"{curr_out_path}/{model_name}_{target_days[i]}.csv")
    print(f"Quantile prediction for day {day} Ended!")


if __name__ == "__main__":
    # generate_quantile_prediction_day(192,type='case',normpop=True,add_shapelet=False,median_first=True)
    # for normpop in [True,False]:
    #     for add_shapelet in [True,False]:
    #         for type in ['case','death']:
    #             generate_quantile_prediction(178,1095,type,add_shapelet=add_shapelet,normpop=normpop,median_first=True)
    # generate_quantile_prediction(178, 1095, 'case', add_shapelet=False, normpop=True, median_first=True)
    # generate_quantile_prediction(178, 1095, 'death', add_shapelet=False, normpop=True, median_first=True)
    #
    metrics = ['WIS','MAE','Coverage']
    for type in ['case','death']:
        for metric in metrics:
            plot_metric(type=type,metric=metric,savefig=True)

    # convert_forecasthub_to_median_format('Data/Raw','Data/Cleaned/State_Death','death')
    # convert_quantile_to_result_format('COVIDhub-trained_ensemble','result/Quantiles/death',type='death')
    # convert_quantile_to_result_format('USC-SI_kJalpha')

    # generate_quantile_prediction(add_shapelet=False)
    # generate_quantile_prediction()
    # generate_quantile_prediction_day(day=192,add_shapelet=True)


    # generate_quantile_prediction_day(day=192)
    # get_all_model_predictions_week(381)
    # convert_forecasthub_to_quantile_format('Data/Raw','Data/Cleaned/State_Death_Quantile','death')
    # compute_dist(178,703)
    # convert_truth()
    # get_all_model_predictions()
    # compute_dist_day(381)
    # get_all_predictions_in_period_dict(227,6,shapelet=True)
   # get_ground_truth(365)
   #  run_sim_range(193,700)
   #  run_sim(381)
   #  error_plot(error=True)
   #  print(get_ground_truth_array([269,276,283,290]))
    # print(get_4weeks_ahead_ground_truth(269))
    # print(get_all_model_predictions())
