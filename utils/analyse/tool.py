import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import math
from sklearn.cluster import KMeans


def read_data(df):
    """
       Read and preprocess the data.

       Parameters:
       df (pd.DataFrame): The input DataFrame.

       Returns:
       tuple: A tuple containing the preprocessed DataFrame, number of GSMs, grouped DataFrames, number of groups, and group names.
    """
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(df.columns[1], axis=1)
    groups = df.iloc[0, 1:]
    df = df.iloc[1:]
    df.set_index(df.columns[0], inplace=True)
    df = df.astype('float')
    df = df.dropna()
    df = df[~(df == 0).any(axis=1)]

    num_gsm = df.shape[0]
    group_arr = []
    group_columns = [[] for _ in range(num_gsm)]
    for i in range(len(groups)):
        if group_arr:
            for index, g_name in enumerate(group_arr):
                if groups.iloc[i] == g_name:
                    group_columns[index].append(df.columns[i])
                    break
                if index == len(group_arr) - 1:
                    group_arr.append(groups.iloc[i])
                    group_columns[index + 1].append(df.columns[i])
                    break
        else:
            group_arr.append(groups.iloc[i])
            group_columns[0].append(df.columns[i])
    group_num = len(group_arr)

    group_df = [[] for _ in range(num_gsm)]
    for j in range(group_num):
        group_df[j] = df[group_columns[j]]

    return df, num_gsm, group_df, group_num, group_arr


def ench_read_data(df):
    """
        Enhanced read and preprocess the data.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        tuple: A tuple containing the preprocessed DataFrame, number of GSMs, grouped DataFrames, number of groups, and group names.
    """
    df = df.drop(df.columns[1], axis=1)
    df = df.drop(df.columns[1], axis=1)
    groups = df.iloc[0, 1:]
    df = df.iloc[1:]
    df.set_index(df.columns[0], inplace=True)
    df = df.astype('float')
    df = df.dropna()
    df = df[~(df == 0).any(axis=1)]

    num_gsm = df.shape[0]
    group_arr = []
    group_columns = [[] for _ in range(num_gsm)]
    for i in range(len(groups)):
        if group_arr:
            for index, g_name in enumerate(group_arr):
                if groups.iloc[i] == g_name:
                    group_columns[index].append(df.columns[i])
                    break
                if index == len(group_arr) - 1:
                    group_arr.append(groups.iloc[i])
                    group_columns[index + 1].append(df.columns[i])
                    break
        else:
            group_arr.append(groups.iloc[i])
            group_columns[0].append(df.columns[i])
    group_num = len(group_arr)

    group_df = [[] for _ in range(num_gsm)]
    for j in range(group_num):
        group_df[j] = df[group_columns[j]]

    return df, num_gsm, group_df, group_num, group_arr


def calculate_p_fc(df1, df2, num_gsm):
    """
    Calculate p-values and fold changes between two conditions.

    Parameters:
    df1 (pd.DataFrame): The first condition DataFrame.
    df2 (pd.DataFrame): The second condition DataFrame.
    num_gsm (int): The number of GSMs.

    Returns:
    pd.DataFrame: A DataFrame containing the mean values, p-values, and log2 fold changes.
    """
    p_arr = []
    if_int = True
    for index in range(5):
        num = df1.iloc[index, 1]
        if int(num) != num:
            if_int = False

    for index in range(num_gsm):
        data1 = df1.iloc[index, :]
        data2 = df2.iloc[index, :]
        t_statistic, p_value = ttest_ind(data1, data2)
        p_value = -math.log10(p_value)
        p_arr.append(p_value)

    condition1 = df1.mean(axis=1)
    condition2 = df2.mean(axis=1)
    new_df = pd.DataFrame(np.array([condition1, condition2, p_arr]).T, columns=['condition1', 'condition2', 'log10(p)'],
                          index=df1.index)

    if if_int:
        new_df['log2FC'] = np.log2(new_df['condition2'] / new_df['condition1'])
    else:
        new_df['log2FC'] = np.log2(new_df['condition2'] + 1) - np.log2(new_df['condition1'] + 1)
        # 然后，删除包含NaN的行
    new_df = new_df[np.all(np.isfinite(new_df), axis=1)]
    new_df = new_df.dropna()
    return new_df


def cluster_calculate(df, n_clusters):
    """
    Perform K-means clustering on the data.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n_clusters (int): The number of clusters.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df['Cluster'] = clusters
    return df


def get_signpoint(df1, df2, num_gsm):
    """
        Identify significant points based on p-values and fold changes.

        Parameters:
        df1 (pd.DataFrame): The first condition DataFrame.
        df2 (pd.DataFrame): The second condition DataFrame.
        num_gsm (int): The number of GSMs.

        Returns:
        pd.DataFrame: A DataFrame containing the significant points.
    """
    new_df = calculate_p_fc(df1, df2, num_gsm)
    data_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    data_df = pd.merge(data_df, new_df, left_index=True, right_index=True, how='inner')

    alpha_threshold = -np.log10(0.05)  # 阈值
    # 筛选出符合条件的点
    significant_points = data_df[
        (data_df['log10(p)'] > alpha_threshold) & (
                (data_df['log2FC'] > 1) | (data_df['log2FC'] < -1))].copy()
    return significant_points


def get_point(df1, df2, num_gsm):
    """
        Classify points based on p-values and fold changes.

        Parameters:
        df1 (pd.DataFrame): The first condition DataFrame.
        df2 (pd.DataFrame): The second condition DataFrame.
        num_gsm (int): The number of GSMs.

        Returns:
        tuple: A tuple containing significant points, non-significant points, and points within the threshold.
    """
    calculate_df = calculate_p_fc(df1, df2, num_gsm)
    alpha_threshold = -np.log10(0.05)  # Threshold
    significant_points = calculate_df[
        (calculate_df['log10(p)'] > alpha_threshold) & ((calculate_df['log2FC'] > 1) | (calculate_df['log2FC'] < -1))]
    non_significant_points = calculate_df[
        (calculate_df['log10(p)'] < alpha_threshold) & ((calculate_df['log2FC'] > 1) | (calculate_df['log2FC'] < -1))]
    point = calculate_df[
        (calculate_df['log10(p)'] > alpha_threshold) & (calculate_df['log2FC'] > -1) & (calculate_df['log2FC'] < 1)]
    return significant_points, non_significant_points, point, calculate_df


def benjamini_hochberg(p_values):
    """
    Perform Benjamini-Hochberg correction for multiple hypothesis testing.

    Parameters:
    p_values (array-like): List or numpy array of p-values to correct.

    Returns:
    numpy array: Array of adjusted p-values (FDR).
    """
    p_values = np.asarray(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    adjusted_p_values = np.zeros(n)

    for i in range(n):
        rank = i + 1
        adjusted_p_values[i] = sorted_p_values[i] * n / rank

    # Ensure adjusted p-values are monotonic
    for i in range(n - 2, -1, -1):
        adjusted_p_values[i] = min(adjusted_p_values[i], adjusted_p_values[i + 1])

    # Place adjusted p-values in original order
    original_order = np.argsort(sorted_indices)
    adjusted_p_values = adjusted_p_values[original_order]

    # Clip adjusted p-values to 1
    adjusted_p_values = np.clip(adjusted_p_values, 0, 1)

    return adjusted_p_values
