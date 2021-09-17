import pandas as pd
import numpy as np
from typing import List
from experiment import Experiment
import math
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


def assign_ccs_to_bins(df, according_to_ccs, bin_size):
    """
    @df: pandas dataframe
    @according_to_css: the css value according to which the bins are assigned
    @bin_size: skalar representing size  of each bin
    @return: df with additional "binned" column that tells to which bin a row was assigned to
    """

    ccs_arr = df[according_to_ccs].values

    bin_start = math.floor(ccs_arr.min())
    bin_end = math.ceil(ccs_arr.max())
    bin_end += bin_end % bin_size + 1
    bins = np.arange(bin_start, bin_end, step=bin_size)
    df.loc[:, "binned"] = np.digitize(ccs_arr, bins)
    return df


def binwise_medians(df, ccs_col, diff_ccs_col):
    """
    out of every feature in each bin get the mean ccs value and the median difference
    to the median feature ccs value out of all experiments
    @df: input pandas dataframe
    @ccs_col: column in df where ccs values are located
    @diff_ccs_col: column where difference between ccs_col values (1st experiment) and feature median
    across experiments is located
    @return: aggregated df pandas dataframe according to "binned" column with median diff 
    """
    df.dropna(subset=[ccs_col, diff_ccs_col], inplace=True)
    return df.groupby(
        by=["binned"]).agg(ccs=(ccs_col, 'mean'), diff_ccs=(diff_ccs_col, 'median')).sort_index()


def chargewise_binning(df, ccs_col, bin_size, diff_ccs_col):
    """
    for each bin with size bin_size across column ccs_col in pandas dataframe df, calculate the median
    of diff_ccs_col for each charge state
    @df: pandas dataframe
    @bin_size:  size of each bin
    @diff_ccs_col: column used to calculate median of
    @ccs_col: column used for bin assignment of rows in df
    @return: list of pandas dataframes with length 3 (one df for each charge-state)
    with each row containing (bin, mean_feat_median_in_bin, median_diff_ccs_to_feat_median_in_bin)
    and columns (bin, ccs, diff_ccs)
    """
    charges = range(2, 5)
    df_charges = [df.loc[df.charge == z].copy() for z in charges]
    df_charges = [assign_ccs_to_bins(
        df_one_charge, ccs_col, bin_size) for df_one_charge in df_charges]
    result_df_list = [binwise_medians(
        df_charges, ccs_col, diff_ccs_col) for df_one_charge in df_charges]
    return result_df_list


def get_outlier_idx(series, scale_factor=1):
    """"
    gets row names of outliers
    @return: list of row names (pandas indices) of outliers in df
    """

    scaled_sd = series.values.std() * scale_factor
    mean = series.values.mean()
    return series[(series > (mean + scaled_sd)) | (series < (mean - scaled_sd))].index


def remove_outlier(df, col, scaling=1):
    drop_idx = get_outlier_idx(df[col], scaling)
    return df.drop(drop_idx)


def align_ccs(df, df_name, funcs_dict, col_to_align):
    """aligns ccs values of one experiment with functions in funcs_dict
    @df:
    @df_name:
    @funcs_dic: dict of  {dataframe_name : {charge state : function}}. function takes a ccs value and returns 
    the correction it has to add in order to be aligned
    @col_to_align: name of column which values to change. usually ccs values"""
    for z, dfunc in funcs_dict[df_name].items():
        df.loc[df.charge == z, col_to_align] = df.loc[df.charge == z, col_to_align].map(
            lambda ccs: ccs + funcs_dict[df_name][z](ccs))
    return df


def add_feat_median(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, "feat_median"] = df.apply(
        np.nanmedian, axis=1, raw=True)
    return df


def calc_diffs_to_feat_median(joined_tables: pd.DataFrame):
    ccs_col_list = [
        col_name for col_name in joined_tables.columns if col_name.startswith("ccs")]
    for ccs_col in ccs_col_list:
        joined_tables.loc[:, "diff_" +
                          ccs_col] = joined_tables.feat_median - joined_tables[ccs_col]
    return joined_tables.reset_index(drop=False)


def average_moving_window_over_diff_ccs(diff_ccs: np.ndarray, filter_size: int):
    return uniform_filter1d(diff_ccs, filter_size, mode="constant")


def get_data_for_lin_interpol(joined_tables, diff_ccs_col, bin_size=5):

    # should look like [df2.0, df3.0, df4.0]
    # with   (bin, mean_of_median_ccs_in_bin, median_diff_ccs_to_feat_median_in_bin) for each row
    """prepares feat_median and diff_ccs for linear intepolation
    @return: list of df """
    binned_medians_list = chargewise_binning(joined_tables, ccs_col="feat_median",
                                             bin_size=bin_size, diff_ccs_col=diff_ccs_col)

    binned_medians_list[:] = [remove_outlier(binned_median_df, "diff_ccs", scaling=2)
                              for binned_median_df in binned_medians_list]

    # smooth diff_ccs vals and replace old values
    smoothed_diff_ccs = [average_moving_window_over_diff_ccs(
        df.diff_ccs.values) for df in binned_medians_list]
    for i in range(len(binned_medians_list)):
        binned_medians_list[i].loc[:, "diff_ccs"] = smoothed_diff_ccs[i]

    return binned_medians_list


def learn_ccs_correction(df: pd.DataFrame, ex_name: str) -> dict:
    """learn correction curve for each charge state of one experiment
    @df: outer-join of experiment data
    @return: {experiment_name: {charge_state: align_function}}
    """
    funcs_dic = {ex_name: dict()}

    diff_ccs_col = "diff_ccs_"+ex_name

    df_charge_list = get_data_for_lin_interpol(df, diff_ccs_col)

    for z_state, df_charge in enumerate(df_charge_list):

        x = df_charge.ccs.values
        y = df_charge.diff_ccs.values
        x_idx = np.argsort(x)
        x = x[x_idx]
        y = y[x_idx]

        interp = interp1d(x, y, kind="linear",
                          fill_value=0, bounds_error=False)

        funcs_dic[ex_name][z_state+2] = interp
    print("is funcs_dic 3 items long per experiment?",
          len(funcs_dic[ex_name]) == 3)
    return funcs_dic


def align_experiements(coll_exs: List[Experiment]) -> List[Experiment]:
    # apply alignment only on unimodal and main feats
    df_names = [ex.name for ex in coll_exs]
    dfs = [exp.select_uni_and_main for exp in coll_exs]
    dfs = [df.loc[:, "ccs"].to_frame() for df in dfs]

    # change names of overlapping column names to prepare for join
    dfs = [df.rename(columns={"ccs": "ccs_"+str(tag)}, errors="raise")
           for df, tag in zip(dfs, df_names)]

    # build the median array across experiments to which they are aligned to
    joined_tables = pd.DataFrame().join(dfs, how="outer")
    joined_tables = add_feat_median(joined_tables)

    # calculate differences of feat ccs of each dataset to respective median
    joined_tables = calc_diffs_to_feat_median(joined_tables)

    # merge the dictionaries
    func_dic = {**learn_ccs_correction(joined_tables, ex_name) for ex_name in df_names}
    # align ccs of each experiment
    result = [Experiment._from_whole_DataFrame(ex.name, align_ccs(
        ex.data, ex.name, func_dic, "ccs")) for ex in coll_exs]
    return result


def intrinsic_align(self):
    pass
