import os
import pandas as pd
import numpy as np
from .data import get_ground_truth_array
from .scoring import *
from .config import *
import matplotlib.pyplot as plt
import seaborn as sns
from .util import *
from datetime import datetime
from datetime import timedelta
import re
sns.set()
def plot_metric(type,metric='WIS',savefig=True):
    def sort_by_last_number(name):
        match = re.search(r'\d+$', name)
        if match:
            return int(match.group())
        return name

    # eval_model = "RandomForestQuantileRegressor_shapelet"
    # eval_model = "RandomForestQuantileRegressorDepth=5_shapelet"
    eval_model = "RandomForestQuantileRegressorNormpop_shapelet"
    # ref_models = ["COVIDhub-trained_ensemble","RandomForestQuantileRegressor","USC-SI_kJalpha"]
    # ref_models = ["COVIDhub-trained_ensemble", "RandomForestQuantileRegressor"]
    # ref_models = ["COVIDhub-trained_ensemble", "RandomForestQuantileRegressorDepth=5","RandomForestQuantileRegressor_shapelet"]
    ref_models = ["COVIDhub-trained_ensemble","RandomForestQuantileRegressor_shapelet","RandomForestQuantileRegressorNormpop"]
    if type == 'case':
        source_dir = output_dir_case_quantile
    elif type == 'death':
        source_dir = output_dir_death_quantile

    eval_path = f"{source_dir}/{eval_model}"
    WIS_eval = {eval_model:{i:[] for i in range(4)}}
    WIS_eval_containing_dates = {eval_model:{i:[] for i in range(4)}} # Used for calculate average
    WIS_eval_dates = {eval_model:{i:[] for i in range(4)}}
    WIS_refs = {ref_model:{i:[] for i in range(4)} for ref_model in ref_models}
    WIS_refs_containing_dates = {ref_model:{i:[] for i in range(4)} for ref_model in ref_models}
    WIS_refs_dates = {ref_model:{i:[] for i in range(4)} for ref_model in ref_models}
    rslt_days = {i:[] for i in range(4)}
    for path_day in sorted(os.listdir(eval_path),key=sort_by_last_number):
        eval_day = int(path_day.split('_')[-1])
        for i in range(4):
            target_day = eval_day + i * 7

            # Load both eval and ref data and find quantiles and states -----------------------

            curr_rslt_eval = pd.read_csv(f"{eval_path}/{path_day}/{eval_model}_{target_day}.csv"
                                         ,index_col=[0]).sort_index()
            curr_rslt_eval.columns = np.round(curr_rslt_eval.columns.astype(float),3)
            curr_rslt_refs = {ref_model:pd.read_csv(f"{source_dir}/{ref_model}/"
                                                    f"{ref_model}_{eval_day}/{ref_model}_{target_day}.csv",
                                                    index_col=[0]).sort_index()
                              for ref_model in ref_models
                              if os.path.exists(f"{source_dir}/{ref_model}/{ref_model}_{eval_day}")}

            all_quantiles = set()
            all_states = set()
            all_dates = set()
            all_quantiles.update(curr_rslt_eval.columns)
            all_states.update(curr_rslt_eval.index)

            for curr_rslt_ref in curr_rslt_refs.values():
                curr_rslt_ref.columns = np.round(curr_rslt_ref.columns.astype(float), 3)
                all_quantiles = all_quantiles.intersection(set(curr_rslt_ref.columns))
                all_states = all_states.intersection(set(curr_rslt_ref.index))

            all_states = sorted(list(all_states))
            all_quantiles = sorted(list(all_quantiles))


            # Update eval and ref to contain the same states and quantiles -----------------------
            curr_rslt_eval = curr_rslt_eval.loc[all_states][all_quantiles]
            for key,value in curr_rslt_refs.items():
                curr_rslt_refs[key] = curr_rslt_refs[key].loc[all_states][all_quantiles]


            ground_truth = get_ground_truth_array([target_day],type=type)
            ground_truth = ground_truth.loc[all_states]

            curr_rslt_eval_dict = curr_rslt_eval.to_dict('list')
            for k,v in curr_rslt_eval_dict.items():
                curr_rslt_eval_dict[k] = np.array(v) # Convert to qdict format required by WIS calculation

            curr_rslt_ref_dicts = {}
            for key,value in curr_rslt_refs.items():
                curr_rslt_ref_dicts[key] = curr_rslt_refs[key].to_dict('list')
                for key2,value2 in curr_rslt_ref_dicts[key].items():
                    curr_rslt_ref_dicts[key][key2] = np.array(curr_rslt_ref_dicts[key][key2])
            # q_left = [round(alpha/2,3) for alpha in all_quantiles]
            # q_right = [:round(1-alpha/2,3) for alpha in all_quantiles]
            # alpha_dict = {round(alpha/2,3):round(1-alpha/2,3) for alpha in all_quantiles}
            # for i in np.arange(0,1,0.01):
            #     try:
            #         metric_eval_curr = weighted_interval_score(ground_truth.values, [], curr_rslt_eval_dict)[
            #             0].mean()
            #         print(i)
            #     except:
            #         continue
            if metric.upper() == 'WIS':
                curr_alphas = [round(1-(all_quantiles[-i-1]-all_quantiles[i]),2) for i in range(len(all_quantiles)//2)]
                metric_eval_curr = weighted_interval_score(ground_truth.values,curr_alphas,curr_rslt_eval_dict)[0].mean()
            elif metric.upper() == 'MAE':
                metric_eval_curr = mean_absolute_error(np.array(ground_truth).flatten(),curr_rslt_eval_dict[0.5])
            elif metric.lower() == 'coverage':
                metric_eval_curr = 1-outside_interval(np.array(ground_truth).flatten(), curr_rslt_eval_dict[0.1],
                                                   curr_rslt_eval_dict[0.9]).mean()

            WIS_eval[eval_model][i].append(metric_eval_curr)
            WIS_eval_containing_dates[eval_model][i].append((target_day,metric_eval_curr))
            WIS_eval_dates[eval_model][i].append(target_day)

            for k,curr_rslt_ref_dict in curr_rslt_ref_dicts.items():
                if metric.upper() == 'WIS':
                    metric_ref_curr = weighted_interval_score(ground_truth.values,curr_alphas,curr_rslt_ref_dict)[0].mean()
                elif metric.upper() == 'MAE':
                    metric_ref_curr = mean_absolute_error(np.array(ground_truth).flatten(), curr_rslt_ref_dict[0.5])
                elif metric.lower() == 'coverage':
                    metric_ref_curr = 1 - outside_interval(np.array(ground_truth).flatten(),
                                                           curr_rslt_ref_dict[0.1],curr_rslt_ref_dict[0.9]).mean()

                WIS_refs[k][i].append(metric_ref_curr)
                WIS_refs_containing_dates[k][i].append((target_day, metric_ref_curr))
                WIS_refs_dates[k][i].append(target_day)

    # Get common dates avaliable for all models to calculate a fair mean
    all_dates = [[] for i in range(4)]
    for inner_dict in WIS_eval_dates.values():
        for k,v in inner_dict.items():
            all_dates[k].append(v)
    for inner_dict in WIS_refs_dates.values():
        for k, v in inner_dict.items():
            all_dates[k].append(v)
    for i in range(len(all_dates)):
        all_dates[i] = set.intersection(*map(set, all_dates[i]))

    date_obj = pd.to_datetime('2020-1-23')


    for i in range(4):
        fig,ax = plt.subplots(figsize=(8,7))
        dates = [date_obj + timedelta(days=i) for i in WIS_eval_dates[eval_model][i]]
        plot_line = ax.plot(dates,WIS_eval[eval_model][i], '-', label=eval_model,markersize=3,alpha=0.5)
        common_mean = np.array([tup[1] for tup in WIS_eval_containing_dates[eval_model][i] if tup[0] in all_dates[i]]).mean()
        ax.axhline(y=common_mean, color=plot_line[0].get_color(), linestyle='--')

        ax.scatter(dates, WIS_eval[eval_model][i], marker='o', s=16, alpha=1.0)
        for ref_model in WIS_refs.keys():
            dates = [date_obj + timedelta(days=i) for i in WIS_refs_dates[ref_model][i]]
            plot_line = ax.plot(dates, WIS_refs[ref_model][i], '-', label=ref_model,markersize=3,alpha=0.5)
            common_mean = np.array([tup[1] for tup in WIS_refs_containing_dates[ref_model][i] if tup[0] in all_dates[i]]).mean()
            ax.axhline(y=common_mean,color=plot_line[0].get_color(), linestyle='--')
            ax.scatter(dates, WIS_refs[ref_model][i], marker='o', s=16, alpha=1.0)
            ax.tick_params(axis='x',labelrotation=20)
            ax.yaxis.set_tick_params(labelsize=13)
        plt.title(f"{i+1} weeks ahead {metric} Score")
        plt.legend()
        plt.xlabel("Days")
        if metric.upper() == 'WIS':
            plt.ylabel(f"WIS Score")
        elif metric.lower() == 'coverage':
            plt.ylabel(f"Coverage")
            plt.ylim([0,1])
        elif metric.upper() == 'MAE':
            plt.ylabel(f"Mean Absolute Error")

        plot_folder = folder(f"result/plots/{type}/{eval_model}")
        if savefig:
            plt.savefig(f"{plot_folder}/{metric}_{i}.png",dpi=300)
        plt.show()





