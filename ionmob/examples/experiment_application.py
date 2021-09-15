import pandas as pd
from ionmob.alignment import experiment

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
ex1 = Experiment(ex_name, seq, charge, ccs,
                 intensity, mz, raw_file, evidence_id)

# or rather like this. be aware of the order of args!
args = df["Modified sequence"].values, df["Charge"].values, df["CCS"].values, df[
    "Intensity"].values, df["m/z"].values, df["Raw file"].values, df["id"].values
ex1 = Experiment(ex_name, *args)

# access the name and data of Experiment like this
print("name of your experiment: ", ex1.name)
print("data of your experiment: ", ex1.data)


# ex1.data itself is a pd.DataFrame so you can use the pandas library to work on it or isolate information from
ex1.data.loc[ex1.data.charge == 2]

# ex1.data is aggregated rows of duplicate features (duplicates of (sequence, charge, ccs)) already, making those unique
# to further aggregate rows and theirby getting rid of possible feature divergence by measurement divergence of ccs values, assign a modality class to each feature
ex2 = ex1.assign_modalities()

# from this point on you can proceed with the inte-experimental CCS alignment
# ....to be continued
