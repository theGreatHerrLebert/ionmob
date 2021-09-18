import pandas as pd
from ionmob.alignment import experiment as exp
from ionmob.alignment import alignment as alig

data_dir = "data/raw_data/"
fname = "M210115_00[1,2,3]_HeLa_grad110_ramp100__evidence.txt"

path = data_dir + fname
df = pd.read_csv(path, sep="\t")

# 2 ways to construct an Experiment object
# 1st methods: isolate columns needed from df (see below) as numpy arrays and pass to constructor

seq, charge, ccs, intensity, mz, raw_file, evidence_id = df["Modified sequence"].values, df["Charge"].values, df[
    "CCS"].values, df["Intensity"].values, df["m/z"].values, df["Raw file"].values, df["id"].values
# give your experiment instance a name. ideally a short uinique version of fname
ex_name = "HeLa_grad110"
ex1 = exp.Experiment(ex_name, seq, charge, ccs,
                     intensity, mz, raw_file, evidence_id)

# or rather like this. be aware of the order of args!
args = df["Modified sequence"].values, df["Charge"].values, df["CCS"].values, df[
    "Intensity"].values, df["m/z"].values, df["Raw file"].values, df["id"].values
ex1 = exp.Experiment(ex_name, *args)

# 2nd method: if you are sure that the output table contains the columns "Modified sequence",
# "Charge", "CCS", "Intensity", "m/z", "Raw file", "id", "Mass", "Number of isotopic peaks",
# "Retention time", "Retention length" ( which is usually the case for MaxQuant evidence.txt),
#  then you can also use this method

ex1 = exp.Experiment.from_MaxQuant_DataFrame(df, "HeLa_grad110")

# access the name and data of Experiment like this
print("name of your experiment: ", ex1.name)
print("data of your experiment: ", ex1.data)


# ex1.data itself is a pd.DataFrame so you can use the pandas library to work on it or isolate information from
ex1.data.loc[ex1.data.charge == 2]

# ex1.data is aggregated rows of duplicate features (duplicates of (sequence, charge, ccs)) already, making those unique
# to further aggregate rows and theirby getting rid of possible feature divergence by measurement divergence of ccs values, assign a modality class to each feature
ex2 = ex1.assign_modalities()

# from this point on you can proceed with the inter-experimental CCS alignment of experiment
# data aquired by the same device

data_dir = "data/raw_data/"
file_names = ["M210115_00[1,2,3]_HeLa_grad110_ramp100__evidence.txt",
              "M210115_00[4,5,6]_HeLa_grad47_ramp100__evidence.txt",
              "M210115_00[7,8,9]_HeLa_grad20_ramp100__evidence.txt"]
exp_names = ["HeLa_grad110", "HeLa_grad47", "HeLa_grad20"]
paths = [data_dir + fname for fname in file_names]
dfs = [pd.read_csv(path, sep="\t") for path in paths]
exs = [exp.Experiment.from_MaxQuant_DataFrame(
    df, exp_name) for exp_name, df in zip(exp_names, dfs)]
exs = [ex.assign_modalities() for ex in exs]

# perform the ccs alignment of the experiments to each other
aligned_exs = alig.align_experiments(exs)
# merge the aligned
aligned_ex = alig.merge_experiments(aligned_exs)

# if you want to expand your aquired data you can align a dataset aquired by another lab to the first one
# first read and intrinsically align the experiments of the other dataset
data_dir2 = "data/mann_data/"
file_names2 = ["Results_evidence_mann_Drosophila.txt",
               "Results_evidence_mann_HeLaTryp.txt",
               "Results_evidence_mann_Celegans.txt"]
exp_names2 = ["mann_Drosophila", "mann_HeLaTryp", "mann_Celegans"]
paths2 = [data_dir2 + fname for fname in file_names2]
dfs2 = [pd.read_csv(path, sep="\t") for path in paths]
exs2 = [exp.Experiment.from_MaxQuant_DataFrame(
    df, exp_name) for exp_name, df in zip(exp_names2, dfs2)]
exs2 = [ex.assign_modalities() for ex in exs2]

aligned_exs2 = alig.align_experiments(exs2)

aligned_ex2 = alig.merge_experiments(aligned_exs2)
