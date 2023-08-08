import pandas as pd
import numpy as np
from typing import List, Callable, Dict
from ionmob.preprocess.experiment import Experiment
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
        df_one_charge, ccs_col, diff_ccs_col) for df_one_charge in df_charges]
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
    return df.copy()


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


def average_moving_window_over_diff_ccs(diff_ccs: np.ndarray, filter_size: int = 11):
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
    @df: outer-join of experiment example_data
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
    return funcs_dic


def align_experiments(coll_exs: List[Experiment]) -> List[Experiment]:
    # apply alignment only on unimodal and main feats
    df_names = [ex.name for ex in coll_exs]
    keys = ["sequence", "charge"]
    dfs = [exp.select_uni_and_main().set_index(keys) for exp in coll_exs]
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
    func_dic = {}
    for ex_name in df_names:
        func_dic.update(learn_ccs_correction(joined_tables, ex_name))
    # align ccs of each experiment
    result = [Experiment._from_whole_DataFrame(ex.name, ex.int_to_raw, align_ccs(
        ex.data, ex.name, func_dic, "ccs")) for ex in coll_exs]
    return result


def agg_feats_after_merge(df: pd.DataFrame, ccs_agg_func: Callable[[pd.Series], float]) -> pd.DataFrame:
    """
    aggregates (modified_sequence, charge)-duplicates for dfs with containers in columns. CCS values are
    aggregated according to ccs_agg_func
    @df: df that only contains instances that were labeles as 'main' and 'measurement_error' but do not have
    a corresponding 'secondary' instance and therefore are not actually bimodal
    @cc_agg_func: function to use for aggregation of CCS values of (modified_sequence, charge)-duplicates
    @modality_class: new modality class assigned to all feats in newly aggregated df
    """
    def concat_sets(x): return set().union(*x)
    def get_first(series): return series.iloc[0]

    aggregated_df = df.groupby(by=["sequence", "charge"]).agg(
        intensities=("intensities", "sum"), feat_intensity=("feat_intensity", "sum"),
        occurences=("occurences", "sum"), raw_files=("raw_files", concat_sets),
        #ids=("ids", "sum"),
        ccs=("ccs", ccs_agg_func), mz=("mz", get_first),
        rt_min=("rt_min", "sum"), rt_max=("rt_max", "sum"),
        mz_min=("mz_min", "sum"), mz_max=("mz_max", "sum")
    ).reset_index(drop=False)
    return aggregated_df


def merge_experiments(coll_exs: List[Experiment], new_name: str) -> Experiment:
    dfs = [ex.data.copy() for ex in coll_exs]

    def get_combined_dict_of_exs():
        """combines the int_to_raw members of Experiment to form new int_to_raw dict"""
        raw_file_names = []
        for ex in coll_exs:
            for k, raw_file_name in ex.int_to_raw.items():
                raw_file_names.append(raw_file_name)
        combined_raw_file_names = list(set(raw_file_names))
        return dict(zip(combined_raw_file_names, range(len(combined_raw_file_names))))

    comb_raw_to_int = get_combined_dict_of_exs()

    def renew_encoding_raw_files_col(df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:

        def change_to_new_encoding(old_code: set, old_int_to_raw: Dict[int, str]) -> set:
            """changes the encoding for raw files of respective int_to_raw member
            into new encoding according to combined dict of all experiment int_to_raw members"""
            return {comb_raw_to_int[old_int_to_raw[i]] for i in old_code}

        # transfer {original_encoded_ints} -> {new_encoding_ints}
        for i in range(len(coll_exs)):
            raw_file_new_int = np.vectorize(change_to_new_encoding)(
                df_list[i].raw_files.values, coll_exs[i].int_to_raw)
            df_list[i].loc[:, "raw_files"] = raw_file_new_int
        return df_list

    dfs = renew_encoding_raw_files_col(dfs)

    merged_df = pd.concat(dfs, ignore_index=True)
    agged_df = agg_feats_after_merge(merged_df, np.mean)

    return Experiment._from_whole_DataFrame(new_name, comb_raw_to_int, agged_df)


def get_chargewise_mean(df: pd.DataFrame, col_name: str) -> Dict[int, np.float64]:
    result_dic = {z: df.loc[df.charge == z, col_name].values.mean() for z in range(2, 5)
                  if not df.loc[df.charge == z, col_name].empty}
    return result_dic


def apply_mean_shift(ref: Experiment, exp: Experiment) -> Experiment:
    """apply a shift on ccs values of exp to correct for experimental or device shifts.
    a chargewise shift is applied on exp by calculating the difference of ccs values
    detected in both experiments and their chargewise mean before applying those means
    on the ccs values of exp depending on the charge state.
    :ref: reference experiment towards which the example_data of exp is corrected
    :exp: experiment that undergoes correction
    :return: Experiment instance that is essentially exp with additional column  "shifted_ccs" of corrected ccs values
    """
    c_pairs = ref.data.merge(exp.data, left_on=['sequence', 'charge'], right_on=[
                             'sequence', 'charge'])
    c_pairs['diff'] = c_pairs['ccs_x'] - c_pairs['ccs_y']
    charge_to_factor = get_chargewise_mean(c_pairs, col_name="diff")

    def apply_shift(number, condition):
        return number + charge_to_factor[condition]

    shifted_df = exp.data.copy()
    shifted_df.loc[:, "shifted_ccs"] = exp.data.apply(
        lambda row: apply_shift(row["ccs"], row["charge"]), axis=1)

    return Experiment._from_whole_DataFrame(exp.name, exp.int_to_raw, shifted_df)


def adopt_shifted_ccs(exp: Experiment) -> Experiment:
    """substitutes ccs values with values of shifted_ccs and removes latter one
    :exp: Experiment with columns ["ccs", "shifted_ccs"]
    :return: Experiment where "shifted_ccs" was adopted to be "ccs"
    """
    df = exp.data.copy().drop(columns="ccs").rename(
        columns={"shifted_ccs": "ccs"})

    return Experiment._from_whole_DataFrame(exp.name, exp.int_to_raw, df)


# alternative method for shift correction by caluculating the shift as
# difference between 2 linear regressions of each charge state and experiment

# def seperate_by_charge(df: pd.DataFrame) -> List[pd.DataFrame]:
#     """split pandas Dataframe according to charge column
#     """
#     result = []
#     for _, g_df in df.groupby(by= ["charge"]):
#         result.append(g_df)

#     return result

# def learn_ccs_mz_linReg(ex: Experiment) -> dict:
#     """learn linear regression line for each charge state of an experiment
#     ccs VS mz values
#     @ex: Experiment instance on which ccs and mz values a linear regression is learned
#     @return: {experiment_name: {charge_state: align_function}}
#     """
#     ex_name = ex.name
#     df = ex.example_data
#     funcs_dic = {ex_name: dict()}

#     diff_ccs_col = "diff_ccs_"+ex_name

#     df_charge_list = seperate_by_charge(df)

#     for z_state, df_charge in enumerate(df_charge_list):

#         x = df_charge.ccs.values
#         y = df_charge.diff_ccs.values
#         x_idx = np.argsort(x)
#         x = x[x_idx]
#         y = y[x_idx]

#         #linreg_func = linreg....

#         funcs_dic[ex_name][z_state+2] = linreg_func
#     return funcs_dic


# def apply_mean_shift2(ref: Experiment, exp: Experiment) -> Experiment:
#     """shift factor anhand von chargewise linReg für ref und exp daten berechnet"""
#     pass
