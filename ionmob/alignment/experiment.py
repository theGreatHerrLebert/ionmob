import pandas as pd
import numpy as np






class Experiment:
    
    # auch noch dic mit ("ptm"->"ptm-token-in-seq") um sequences in Experiments zu standardisieren
    def __init__(self, name : str, seq : np.ndarray, charge : np.ndarray,
                 ccs : np.ndarray, intensity: np.ndarray, mz: np.ndarray,
                 raw_file : np.ndarray, evidence_id: np.ndarray):
        self.raw_dict = dict(zip(set(raw_file), range(len(set(raw_file)))))
        raw_file_int = np.vectorize(self.raw_dict.get, otypes=[int])(raw_file)
        self.name = name
        df = pd.DataFrame({"sequence":seq, "charge":charge, "ccs":ccs,
                                  "intensity":intensity, "mz": mz, "raw_file": raw_file_int,
                                  "id": evidence_id})
        self.data = self._cleanup(df)
        

    def __repr__(self):
        return "Experiment: {}\n".format(self.name) + self.data.__repr__()
    
    def intrinsic_align(self):
        pass
        
    def master_align(self, other):
        pass
    
    @staticmethod    
    def _reduce_dup_feats(df: pd.DataFrame) -> pd.DataFrame:
        """
        aggregates ("modified_sequence", "charge", "ccs")-duplicates
        """
        # calculate total intensity of (seq, z, ccs) duplicates in table (across runs)
        subset=["sequence", "charge", "ccs"]
        # groupby() removes rows with nan values in subset additionally to grouping
        df = df.groupby(by = subset)

        #  agg() agregates and changes column names
        #! maybe instead of list and set use tuples
        df = df.agg(intensities = ("intensity", list),   
                    feat_intensity = ("intensity", "sum"),
                    mz = ("mz", set), occurences = ("sequence", "count"),
                    raw_files = ("raw_file", set), ids = ("id", list)
                   ).reset_index(drop=False)
        print("_reduce_dup_feats():",df.columns)
        return df
    
    @staticmethod
    def _drop_unnecessary_rows(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        df = df.drop(df[df.charge ==1].index)
        print("_drop_unnecessary_rows():",df.columns)
        
        return df
    
    @staticmethod
    def _sort_feats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by = ["sequence", "charge", "ccs"]).reset_index(drop= False)
        print("_sort_feats():",df.columns)
        return df
    
    @classmethod
    def _cleanup(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        removing rows with nan in any column of and singly charged features.
        """
        if not df.shape[0]:
            return df
        df = cls._drop_unnecessary_rows(df)
        df = cls._reduce_dup_feats(df)
        return cls._sort_feats(df)
    
    # @staticmethod
    # def _container_in_cols(df: pd.DataFrame, target_cols):
    #     """
    #     identifies which columns contains container (list, numpy.array, set, dict, tuple) as values
    #     @return: numpy array with bools which column contains container
    #     """
    #     target_cols_idx = [i for i, col_name in enumerate(df.columns) if col_name in target_cols]
    #     mask = np.zeros((len(df.columns)), dtype = bool)
    #     for i, val in enumerate(df.iloc[0, target_cols_idx]):
    #             try:
    #                 # check if val is a container introduced by aggregation 
    #                 if len(val):
    #                     mask[target_cols_idx[i]] = True
    #             except TypeError as err:
    #                 pass
    #     return mask
    @staticmethod
    def add_main_feat_values(df, condition_main):
        """
        adds columns with values of respective main_feature to each row. For each column in target_cols that contains a container data type, a
        seperate colum with name max_feat_{colname} is generated with the max value of respective container 
        :param condition_main: column which determines main feature. currently only "feat_intensity" or "occurences" considered
        :return: dataframe with len(target_columns) additinal columns with "main_" prefix to targeted column name
        """    
        target_col = ["ccs"]

        subset=["sequence", "charge"]
        main_vals_col = np.empty((df.shape[0],))
        main_vals_col.fill(np.nan)
        new_col_name = "main_ccs"
        main_feat_df = pd.DataFrame({new_col_name: main_vals_col}, index = df.index)

        for _, g_df in df.groupby(by = subset):
            idx_main = g_df[condition_main].idxmax()
            main_vals = g_df.loc[idx_main, target_col].values
            main_feat_df.loc[g_df.index] = np.full((g_df.shape[0], main_vals.shape[0]), main_vals)

        return pd.concat([df, main_feat_df], axis= 1)

    @staticmethod
    def calc_diffs_to_main_feat(df, cols, secondary_level = False):
        """
        calculates difference between value in cols and corresponding main_feature value of respective main_feature
        @df: pandas DataFrame object
        @cols: itrable with column names to for which value differences to main_feature are determined
        @secondary_level: for usage on subset. ulitmately adds suffix to additional columns with calculated values
        @return: dataframe with len(cols) additinal columns with "difference_" prefix to column names in cols
        """
        secondary_col = ""
        if secondary_level:
            secondary_col = "_secondary"
        for col in cols:
            if col not in df.columns:
                raise KeyError("Column {} not present in given DataFrame".format(col))
                
        for col in cols:
            main_col = "main_"+ col+secondary_col
            df.loc[:,"difference_"+ col+secondary_col] = df[col] - df[main_col]
        
        return df
    
    @classmethod
    def empty_experiment(cls, name):
        args = [[]]*7
        return cls(name, *args)
    
    @classmethod
    def _from_whole_DataFrame(cls, name, df):
        # instanciate empty experiment and fill mit df
        new_exp = cls.empty_experiment(name)
        new_exp.data = df
        return new_exp
    
    @classmethod
    def from_MaxQuant_DataFrame(cls, df: pd.DataFrame, name: str):
        cls._validate(df)
        return cls(name, df["Modified sequence"].values, df["Charge"].values, df["CCS"].values, df["Intensity"].values, df["m/z"].values, df["Raw file"].values, df["id"].values)

    @staticmethod
    def _validate(obj):
        # verify presence of necessary columns
        cols = ["Modified sequence", "Charge", "m/z", "CCS", "Intensity", "Raw file", "Mass"]
        if not set(cols).issubset(set(obj.columns)):
            raise AttributeError("Must have {}.".format(cols))     
            
    def prep_feat_analysis(self, condition_main):
        """wraps pipeline for preperation for modality class assignment
        @df: pandas DataFrame
        @condition_main: column on which the main feat identification is based
        @return: datframe with additional max_feat_{colname} or
        """
        df = self.add_main_feat_values(self.data, condition_main)
        df = self.calc_diffs_to_main_feat(df, ["ccs"])
        return self._from_whole_DataFrame(self.name, df)
        

    

        
# def read_evidence(path):
#     relevant_cols = ["Modified sequence", "Charge", "m/z", "CCS", "Intensity", "Raw file", "Mass"]
#     df = read_raw_and_filter(relevant_cols, path)
#     validate(df)
#     return from_DataFrame(df)



    