# ionmob
### A Framework for Predicting Collision Cross Section Values of Peptide-Ions with Traditional and Deep Machine Learning Methods

```ionmob``` is a python package for predicting **CCS** values of peptides.
Not only does it contain several pre-trained regression models for this task, but it also introduces a full pipeline to link data preprocessing, model training and CCS value inference.
Models are implemented with recent versions of either [TensorFlow](https://www.tensorflow.org/) or [scikit-learn](https://scikit-learn.org/stable/).
Please feel free to use, alter or extend ```ionmob``` in any way that suits you as it is free and open source under the **GNU General Public License v3.0**.
Feel also free to let us know about missing functionality, bugs, or contributions you would like to make!

* [**ionmob**](#ionmob)
* [**What is a peptide CCS value?**](#what-is-a-peptide-CCS-value)
* [**Why do we measure CCS values of ions?**](#Why-do-we-measure-CCS-values-of-ions)
* [**Why would I want to predict CCS values of peptides in silico?**](#why-would-I-want-to-predict-CCS-values-of-peptides-in-silico)
* [**Can I still use ionmob even if am no machine learning expert?**](#Can-I-still-use-ionmob-even-if-am-no-machine-learning-expert?)
* [**What can I do with ionmob if I am an experienced coder?**](#What-can-I-do-with-ionmob-if-I-am-an-experienced-coder?)
* [**A simple example of CCS prediction and performance evaluation with pre-trained models**](#A-simple-example-of-CCS-prediction-and-performance-evaluation-with-pre-trained-models)
* [**What is a peptide CCS value?**](#what-is-a-peptide-CCS-value?)

---
### What is a peptide CCS value?
The rotationally-averaged collision cross-section - **CCS** - is a scalar value that describes a physical property of an ion with respect to a charge neutral gas.
It can be directly linked to its ion mobility.
The ion mobility is used as an additional separating dimension in high throughput mass spectrometry.
It supplements the measurements of retention times and mass-to-charge ratios and ultimately leads to improved peptide identification.

---
### Why do we measure CCS values of ions?
The CCS value of an ion is a coarse descriptor of its 3D structure.
Since peptides are chains (strings) of amino acids, there exist permutations in nature that have exactly the same mass and chemical properties.
Distinguishing between such peptides with conventional methods like e.g. LC-MS-MS is therefore challenging.
Furthermore, post translational modifications (PTMs) might have only a small impact on an ions mass but alter the functionality of a protein.
Since both a permutation of sequence as well as PTMs have significant impact on 3D structure, one can use ion mobility separation to distinguish between them.
CCS value calculations then give us a measure on how extensively their rotationally-averaged collision cross-section differed.

---
### Why would I want to predict CCS values of peptides in silico?
First, a predictor might give you insight into factors that drive ion mobility.
This information could then be used to optimize your laboratory workflows or uncover yet unknown relationships.
Second, the high reproducibility of measured CCS values in the lab make it an ideal candidate to increase confidence in peptide identifications from database searches.
We think: The recent triumph of ion mobility enhanced mass spectrometry paves the way for expressive predictors by providing previously unavailable amounts of training data!

---
### Can I still use ionmob even if am no machine learning expert?
Definitely yes!
We implemented and pretrained models of different complexity that allow for in silico prediction of CCS values for peptide ions of different charge states out-of-the-box.
They are easily integratable into your existing proteomics workflows.
All you need is a little bit of python scripting experience.
A short introduction can be found down below, or you could have a look at our collection of example notebooks!

---
### What can I do with ionmob if I am an experienced coder??
We made sure that our framework provides a modular set of tools for more experienced coders that want to implement their own models, training strategies or data preprocessing pipelines.
Have a look at our example notebooks for advanced workflow implementation.
Feel also free to contribute any optimizations, models or ideas that you come up with.
This will ultimately help to push prediction accuracy to a point where it provides a huge benefit for rescoring of peptide identifications!

---
### A simple example of ccs prediction and performance evaluation with pre-trained models
Let us assume that you want to have a look at prediction performance for two different ionmob predictors on data of peptide identifications that came from some source.
For ionmob models, you should at least have the following information per peptide: **mz, charge, sequence, ccs**.
CCS values are optional in the general case but are required if you want to compare CCS predictions to CCS measurements.
We will demonstrate how to do this with one of our provided example datasets:

```python
import pandas as pd

# read data and a predictor
data = pd.read_hdf('Tenzer.h5')
data.head()
```

This is what the data looks like:

|       mz |   charge | sequence                                      |   ccs   | origin     |
|---------:|---------:|:----------------------------------------------|--------:|:-----------|
|  801.89  |        2 | \_AAAAAAAAGGAGDSGDAVTK\_                      | 433.825 | Tenzer-lab |
| 1482.86  |        3 | \_AAAAAPASEDEDDEDDEDDEDDDDDEEDDSEEEAMETTPAK\_ | 701.41  | Tenzer-lab |
|  410.205 |        2 | \_AAAACLDK\_                                  | 296.58  | Tenzer-lab |
|  471.28  |        2 | \_AAAAVVAAAAR\_                               | 348.332 | Tenzer-lab |
|  516.27  |        3 | \_AAADALSDLEIKDSK\_                           | 467.791 | Tenzer-lab |

Let's compare accuracy for two predictors.
One that only does a zero-information square-root fit on ion mz values and a deep model that also uses information on peptide sequences.
The latter also needs a so-called [tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer): a tool that translates sequence symbols into a numerical representation.
It is specific for a pretrained model and therefore needs to be loaded as well:

```python
import tensorflow as tf
from matplotlib import pyplot as plt
from ionmob.preprocess.data import sqrt_model_dataset

# read the pretrained predictors
sqrtModel = tf.keras.models.load_model('pretrained-models/SqrtModel')
gruModel = tf.keras.models.load_model('pretrained-models/GRUPredictor/')

# read tokenizer for deep model
tokenizer = tokenizer_from_json('pretrained-models/tokenizer.json')

# create dataset for sqrt prediction and predict
tensorflow_ds_sqrt = sqrt_model_dataset(data.mz, data.charge, data.ccs).batch(1024)
data['ccs_predicted_s'] = sqrtModel.predict(tensorflow_ds_sqrt)

# create dataset for deep prediction and predict
tensorflow_ds_deep = get_tf_dataset(data.mz, data.charge, data.sequence, data.ccs, tokenizer,
                                    drop_sequence_ends=False, add_charge=True).batch(1024)
ccs_predicted_gru, _ = gruModel.predict(tensorflow_ds_deep)
data['ccs_predicted_g'] = ccs_predicted_gru
```

Let's compare prediction accuracies:
```python
import numpy as np

# define error functions
def mean_abs_error(ccs, ccs_pred):
    return np.round(np.mean([np.abs(x[0] - x[1]) for x in np.c_[ccs, ccs_pred]]), 2)

def mean_perc_error(ccs, ccs_pred):
    return np.round(np.mean([np.abs((x[0] - x[1]) / x[0]) * 100 for x in np.c_[ccs, ccs_pred]]), 2)

# show results
print(f"sqrt mean absolute percent error: {mean_perc_error(data.ccs, data.ccs_predicted_s)}")
print(f"gru mean absolute percent error : {mean_perc_error(data.ccs, data.ccs_predicted_g)}")
print("")
print(f"sqrt mean absolute error        : {mean_abs_error(data.ccs, data.ccs_predicted_s)}")
print(f"gru mean absolute error         : {mean_abs_error(data.ccs, data.ccs_predicted_g)}")
```

This then gives us CCS accuracies of:

```python
sqrt mean absolute percent error: 2.58
gru  mean absolute percent error: 1.84

sqrt mean absolute error        : 12.69
gru  mean absolute error        : 9.04
```

Finally, let's visualize the predictions compared to the CCS measurements:

```python
from matplotlib import pyplot as plt

# visualize the charge states in different colors
color_dict = {2:'red', 3:'orange', 4:'lightgreen'}

# create the plot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,4), dpi=200, sharey=True, sharex=True)

ax1.set_title('sqrt fit prediction')
ax1.set_ylabel('CCS')
ax1.set_xlabel('MZ')
ax2.set_xlabel('MZ')
ax2.set_title('deep prediction')

ax1.scatter(data.mz, data.ccs, s=10, alpha=.5, label='ground truth')
ax1.scatter(data.mz, data.ccs_predicted_s, s=10, alpha=.5, c=[color_dict[x] for x in data.charge],
            label='prediction')
ax2.scatter(data.mz, data.ccs, s=10, alpha=.5, label='ground truth')
ax2.scatter(data.mz, data.ccs_predicted_g, s=10, alpha=.2, c=[color_dict[x] for x in data.charge],
            label='prediction')
ax1.legend()
ax2.legend()
fig.show()
```

This code will result in the following plot:

<p align="center">
  <img src="docs/images/sqrt_model.png" width="900" title="prediction vs ground truth">
</p>

---
### Getting insight into driving factors of ion-mobility
Recent papers that worked on ion-mobility prediction such as Chang et al.[^fn2] and Meier et al.[^fn1] identified factors that drive differences in ion mobility.
By using an in silico digest of the human proteome, we will now visit two of them, namely the gravy score and helicality of peptides.
We will start at an initial guess about an ions ccs value, derived from the simple formula:

<img src="https://render.githubusercontent.com/render/math?math=\mathrm{CCS}_{\mathrm{init}}(\mathrm{mz}, c)=s_c\times\sqrt{\mathrm{mz}} %2B b_c">

Where a slope <img src="https://render.githubusercontent.com/render/math?math=s_c">  and an intercept <img src="https://render.githubusercontent.com/render/math?math=b_c"> are fit separately for each modeled charge state <img src="https://render.githubusercontent.com/render/math?math=c">.
The reason why ion-mobility does add an additional dimension of separation is the fact that an ion's CCS value does not always lie on that line.
If it did, CCS would be perfectly correlated with m/z and therefore add no new information.
The idea is now to look at the residues with respect to the square root fit, meaning the vertical difference to our initial guess.
This residue could be provided by any predictor.
We will look at our best performing one: the GRU-based predictor.
It is based on deep GRU-units that can take into account sequence specific higher-order information derived from training data.
We will expand our mathematical formulation of the problem as follows:

<img src="https://render.githubusercontent.com/render/math?math=\mathrm{CCS}_{\mathrm{final}}(\mathrm{mz}, c, s \vert M) = \mathrm{CCS}_{\mathrm{init}}(\mathrm{mz}, c) %2B M(s, \theta)">

Here, a regressor <img src="https://render.githubusercontent.com/render/math?math=M"> (GRU-units) with parameter set <img src="https://render.githubusercontent.com/render/math?math=\theta"> was fit to further lower the mean absolut error (MAE) of predicted CCS values compared to the experimentally observed ones.
For convenience, this predictor does not only return the final predicted ccs value but also the residue with respect to the initial fit, giving us an easy way to link specific features of a given sequence to its impact on ion mobility.
An implementation with ionmob to derive this could look like this:

```python
import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from ionmob.preprocess.helpers import get_gravy_score, get_helix_score, tokenizer_from_json
from ionmob.preprocess.data import get_tf_dataset, sqrt_model_dataset

# read in silico digested human proteome to gain insight into predictors behaviour
data = pd.read_hdf(Synthetic.h5').sample(frac=0.25)

# read predictors and tokenizer
gruModel = tf.keras.models.load_model('pretrained-models/GRUPredictor/')
sqrtModel = tf.keras.models.load_model('pretrained-models/SqrtModel/')
tokenizer = tokenizer_from_json('pretrained-models/tokenizer.json')

# generate tensorflow datasets for prediction
tensorflow_ds_sqrt = sqrt_model_dataset(data.mz, data.charge, None).batch(1024)
tensorflow_ds_deep = get_tf_dataset(data.mz, data.charge, data.sequence, None, tokenizer,
                                    drop_sequence_ends=False, add_charge=True).batch(1024)

# predict with sqrt-fit
ccs_predicted_sqrt = sqrtModel.predict(tensorflow_ds_sqrt)

# predict with deep fit
ccs_predicted_gru, deep_part = gruModel.predict(tensorflow_ds_deep)

# append predictions to dataframe
data['ccs_predicted_gru'] = ccs_predicted_gru
data['ccs_predicted_sqrt'] = ccs_predicted_sqrt
data['ccs_predicted_deep'] = deep_part

# create normalized value of deep increase or decrease prediction of CCS
data['deep_normalized'] = data.ccs_predicted_deep / data.mz

# calculate gravy and helix scores for each sequence
gravy = [get_gravy_score(s, normalize=False) for s in data.sequence]
helix = [get_helix_score(s) for s in data.sequence]

# append calculated values to dataframe
data['gravy'] = gravy
data['helix'] = helix

# select a single charge state to deconvolce differences between charges
charge_2 = data[data['charge'] == 2]
```

We are now ready to have a look at how both gravy score and helix score of a given peptide are correlated with an increase or decrease of the deep predicted ccs with respect to the initial guess. Since the impact is not equal along the mz axis, the deep residue value was normalized by dividing it by the m/z value of its ion. We will calculate the pearson correlation to have some objective measure how strong they are correlated:

```python
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# extract values to correlate
x = charge_2.deep_normalized.values
y_gravy = charge_2.gravy.values
y_helix = charge_2.helix.values

print('Gravy Pearson:', np.round(pearsonr(x, y_gravy), 2))
print('Helix Pearson:', np.round(pearsonr(x, y_helix), 2))
```
This gives us pearson correlation and p values for both gravy and helicality analysis:

```python
Gravy Pearson: [0.46 0.  ]
Helix Pearson: [0.5 0. ]
```

Once again let's visualize this to get a better feel for what the numbers are telling us:

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable

# create the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16,12), dpi=200)

ax1.set_title('linear correlation helicality, mobility')
ax1.set_ylabel('helix score')
ax1.set_xlabel('relative mobility trend')
ax2.set_xlabel('MZ')
ax2.set_title('Deep vs Sqrt prediction')

im1 = ax1.scatter(charge_2.deep_normalized, charge_2.helix, c=charge_2.helix, alpha=.3, s=10, label='data points')
im1 = ax1.scatter(charge_2.deep_normalized, y_line_helix, s=10, c='red', label='linear trend')

im2 = ax2.scatter(charge_2.mz, charge_2.ccs_predicted_gru, s=10, c=charge_2.helix - np.mean(data.gravy), alpha=.3, label='data points')
im2 = ax2.scatter(charge_2.mz, charge_2.ccs_predicted_sqrt, s=2, c='red', alpha=.3, label='sqrt prediction')
ax1.legend()
ax2.legend()

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical', ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['0', '0.5', '1'])

ax3.set_title('linear correlation gravy, mobility')
ax3.set_ylabel('gravy score')
ax3.set_xlabel('relative mobility trend')
ax4.set_xlabel('MZ')
ax4.set_title('Deep vs Sqrt prediction')

im3 = ax3.scatter(charge_2.deep_normalized, charge_2.gravy, c=charge_2.gravy, alpha=.3, s=10, label='data points')
im3 = ax3.scatter(charge_2.deep_normalized, y_line_gravy, s=10, c='red', label='linear trend')

im4 = ax4.scatter(charge_2.mz, charge_2.ccs_predicted_gru, s=10, c=charge_2.gravy, alpha=.3, label='data points')
im4 = ax4.scatter(charge_2.mz, charge_2.ccs_predicted_sqrt, s=2, c='red', alpha=.3, label='sqrt prediction')
ax3.legend()
ax4.legend()

divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(im3, cax=cax, orientation='vertical', ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(['< -4', '0', '> 4'])

fig.show()
```
This code then creates:

<p align="center">
  <img src="docs/images/gravy_helix_linear_correlation.png" width="900" title="prediction vs ground truth">
</p>

As we can observe, our predictor is able to reproduce findings that were already postulated by Chang et al. as well as Meier et al.: Higher gravy and helicality values indeed lead to higher CCS values (at least with respect to our trained predictor). This correlation is by no means perfect, but it lies in the nature of complex interactions that lead to a peptide's 3D structure that they cannot easily be modelled by a simple set of descriptors. Ultimately, this is why a complex function modelling technique like Deep Learning can add something new!

[^fn1]: Deep learning the collisional cross sections of the peptide universe from a million experimental values. Nat Commun, 2021. https://doi.org/10.1038/s41467-021-21352-8
[^fn2]: Sequence-Specific Model for Predicting Peptide Collision Cross Section Values in Proteomic Ion Mobility Spectrometry. Journal of Proteome Research, 2021. https://doi.org/10.1021/acs.jproteome.1c00185
