"""
pemwe_analysis.py

A module for analyzing PEM water electrolyzer (PEMWE) performance data.

Author: Kara Ferner
Date: 9/12/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math
import os
from skimage import io
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid



def get_filenames(directory):
    filenames = []
    for item in os.listdir(directory):
        full_path = os.path.abspath(os.path.join(directory, item))
        if os.path.isfile(full_path) and full_path.endswith(".txt"):
            filenames.append(full_path)
        elif os.path.isdir(full_path):
            filenames.extend(get_filenames(full_path))
    return filenames

def get_filenames_csv(directory):
    filenames = []
    for item in os.listdir(directory):
        full_path = os.path.abspath(os.path.join(directory, item))
        if os.path.isfile(full_path) and full_path.endswith(".csv"):
            filenames.append(full_path)
        elif os.path.isdir(full_path):
            filenames.extend(get_filenames(full_path))
    return filenames

def _convert_abs_time_to_total_seconds(df):
    df['seconds'] = pd.to_datetime(df['time/s'])
    df['seconds'] = (df['seconds'] - df['seconds'].min()).dt.total_seconds() + 0.1
    return df

def _combine_dataframes(dfs_list):
    combined_df = pd.concat(dfs_list)
    return combined_df

def _sort_dataframe_by_time(df):
    df_sorted = df.sort_values(by="time/s")
    return df_sorted

def get_raw_data(filenames):
    dfs_list = []
    for file in filenames:
        df = pd.read_csv(file, sep=r'\t(?!\t$)', engine='python')
        dfs_list.append(df)
    combined_df = _combine_dataframes(dfs_list)
    time_sorted_df = _sort_dataframe_by_time(combined_df)
    rel_time_df = _convert_abs_time_to_total_seconds(time_sorted_df)
    return rel_time_df

def getPolCurve_CP(filenames):
    dfs_list = []
    for file in filenames:
        df = pd.read_csv(file, sep="\t(?!\t$)", engine='python')
        df = _average_data_on_Ns(df)
        dfs_list.append(df)
        
    combined_df = _combine_dataframes(dfs_list)
    time_sorted_df = _sort_dataframe_by_time(combined_df)
    df_reset = time_sorted_df.reset_index(drop=True)
    return df_reset

def getPolCurve_GEIS(filenames):
    dfs_list = []
    for file in filenames:
        df = pd.read_csv(file, sep="\t(?!\t$)", engine='python')
        Ns_end_value = df.loc[df['freq/Hz'].ne(0).iloc[::-1].idxmax(), 'Ns']
        df = df[df['Ns'] <= Ns_end_value]
        #make two dataframes, one of the EIS data to calc HFR and another for current hold data
        df_for_HFR = df[df['freq/Hz']!= 0]
        df_for_IV = df[df['freq/Hz']== 0]

        HFRs = _get_HFR_array_Ns(df_for_HFR)
        df = _average_data_on_Ns(df_for_IV)

        df['HFR'] = HFRs
        dfs_list.append(df)
        
    combined_df = _combine_dataframes(dfs_list)
    time_sorted_df = _sort_dataframe_by_time(combined_df)
    df_reset = time_sorted_df.reset_index(drop=True)

    df_add_HFR = _calc_HFR_free(df_reset)
    df_add_HFR = df_add_HFR.drop(columns=['freq/Hz','Re(Z)/Ohm','-Im(Z)/Ohm', 'time/s','cycle number'],errors='ignore')
    return df_add_HFR

def _average_data_on_Ns(pol_curve_raw_data=pd.DataFrame, num_rows_to_average=100):
    pol_curve_averaged_data = pd.DataFrame(columns=pol_curve_raw_data.columns)
    pol_curve_Ns_grouped = pol_curve_raw_data.groupby("Ns")
    for cycle_num, group in pol_curve_Ns_grouped:
        sorted_group = group.sort_index(ascending=True)
        last_rows = sorted_group.tail(num_rows_to_average)
        for original_col_name in pol_curve_raw_data.columns:
            if original_col_name == "time/s":
                time = pd.to_datetime(last_rows['time/s'])
                pol_curve_averaged_data.loc[cycle_num,original_col_name] = max(time)
            else:
                pol_curve_averaged_data.loc[cycle_num,original_col_name] = last_rows[original_col_name].mean()
    return pol_curve_averaged_data

def _get_HFR_array_Ns(df):
    pol_curve_Ns_grouped = df.groupby("Ns")
    HFR_list_for_given_df = []
    for cycle_num, group in pol_curve_Ns_grouped:
        HFR = _calc_HFR(group)
        HFR_list_for_given_df.append(HFR)
    return np.array(HFR_list_for_given_df)

def _calc_HFR(df):
    real = np.array(df['Re(Z)/Ohm'])
    im = np.array(df['-Im(Z)/Ohm'])
    if len(im) < 10:
        HFR = 0
    else:
        for j in range(len(im)):
            if im[j]<0 and im[j+1]>0:
                HFR = ((real[j+1]+real[j])/2)
                break
            HFR = 0
    return HFR

def _calc_HFR_free(df):
    average_HFR = np.average(df['HFR'])
    df['EiR-Free/V'] = df['<Ewe>/V'] - ((df['<I>/mA']/1000)*average_HFR)
    return df

def split_and_average(df: pd.DataFrame) -> pd.DataFrame:
    #drop Ns
    df = df.drop(columns=['Ns'])

    # Split the dataframe into two halves
    mid_idx = len(df) // 2
    df1 = df.iloc[:mid_idx].reset_index(drop=True)
    df2 = df.iloc[mid_idx:].iloc[::-1].reset_index(drop=True)  # Reverse second half
    
    # Ensure both dataframes have the same length for averaging
    if len(df1) != len(df2):
        raise ValueError("DataFrame has an odd number of rows, making equal splitting impossible.")
    
    # Compute the row-wise average
    averaged_df = (df1 + df2) / 2
    
    # Compute the row-wise standard deviation
    std_dev_df = ((df1 - averaged_df) ** 2 + (df2 - averaged_df) ** 2) ** 0.5
    
    # Rename standard deviation columns
    std_dev_df.columns = [f"{col}_std" for col in std_dev_df.columns]
    
    # Concatenate averaged and standard deviation dataframes
    result_df = pd.concat([averaged_df, std_dev_df], axis=1)
    
    return result_df

