# ionmob
### A Framework for Predicting Collision Cross Section Values of Peptide-Ions with Traditional and Deep Machine Learning Methods

```ionmob``` is a python package for predicting **CCS** values of peptides.
Not only does it contain several pre-trained regression models for this task, but it also introduces a full pipeline to link data preprocessing, model training and CCS value inference.
Models are implemented with recent versions of either [TensorFlow](https://www.tensorflow.org/) or [scikit-learn](https://scikit-learn.org/stable/).
Please feel free to use, alter or extend ```ionmob``` in any way that suits you as it is free and open source under the **GNU General Public License v3.0**.
Feel also free to let us know about missing functionality, bugs, or contributions you would like to make!

* [**What is a peptide CCS value?**](#what-is-a-peptide-CCS-value)
* [**Why do we measure CCS values of ions?**](#why-do-we-measure-CCS-values-of-ions)
* [**Why would I want to predict CCS values of peptides in silico?**](#why-would-I-want-to-predict-CCS-values-of-peptides-in-silico)
* [**Can I still use ionmob even if I am no machine learning expert?**](#can-I-still-use-ionmob-even-if-I-am-no-machine-learning-expert)
* [**What can I do with ionmob if I am an experienced coder?**](#what-can-I-do-with-ionmob-if-I-am-an-experienced-coder)
* [**Installation**](#installation)
* [**A simple example of CCS prediction and performance evaluation with pre-trained models**](#a-simple-example-of-CCS-prediction-and-performance-evaluation-with-pre-trained-models)
* [**Getting insight into driving factors of ion-mobility**](#Getting-insight-into-driving-factors-of-ion-mobility)
* [**Implementing a custom deep CCS predictor**](#Implementing-a-custom-deep-CCS-predictor)

---
### What is a peptide CCS value?
The rotationally-averaged collision cross-section - **CCS** - is a scalar value that describes a physical property of an ion.
It can be directly linked to its ion mobility, meaning its interactive behaviour with respect to a charge neutral gas.
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
### Can I still use ionmob even if I am no machine learning expert?
Definitely yes!
We implemented and pretrained models of different complexity that allow for in silico prediction of CCS values for peptide ions of different charge states out-of-the-box.
They are easily integratable into your existing proteomics workflows.
All you need is a little bit of python scripting experience.
A short introduction can be found down below, or you could have a look at our collection of example notebooks.

---
### What can I do with ionmob if I am an experienced coder?
We made sure that our framework provides a modular set of tools for more experienced coders that want to implement their own models, training strategies or data preprocessing pipelines.
Have a look at our example notebooks for advanced workflow implementation.
Feel also free to contribute any optimizations, models or ideas that you come up with.
This will ultimately help to push prediction accuracy to a point where it provides a huge benefit for rescoring of peptide identifications!

---
### Installation
We recommend to install ionmob into a separate [python virtual environment](https://docs.python.org/3/tutorial/venv.html). Once activated, you can install the ionmob package into it as follows: 
```
git clone https://github.com/theGreatHerrLebert/ionmob.git
cd ionmob
pip install -e .
```

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

You can also try this yourself by cloning this repository and running [this notebook](/notebook/CheckAccuracy.ipynb).

---
### Getting insight into driving factors of ion-mobility
Recent papers that worked on ion-mobility prediction such as Chang et al.[^fn2] and Meier et al.[^fn1] identified factors that drive differences in ion mobility.
By using an in silico digest of the human proteome, we can estimate the impact of two of them, namely the gravy score and helicality of peptides. Our modelling approach will look like this: first an initial CCS value is calculated soley on an ions mass and charge. This is done using the simple formula below:

<img src="https://render.githubusercontent.com/render/math?math=\mathrm{CCS}_{\mathrm{init}}(\mathrm{mz}, c)=s_c\times\sqrt{\mathrm{mz}} %2B b_c">

Where a slope <img src="https://render.githubusercontent.com/render/math?math=s_c">  and an intercept <img src="https://render.githubusercontent.com/render/math?math=b_c"> are fit separately for each modeled charge state <img src="https://render.githubusercontent.com/render/math?math=c">.
The reason why ion-mobility does add an additional dimension of separation is the fact that an ion's CCS value does not always lie on that line.
If it did, CCS would be perfectly correlated with m/z and therefore add no new information.
We can improve our inital CCS prediction modell by also predicting the residues with respect to the square root fit, meaning the vertical difference to our initial value.
These residues could be provided by any predictor but let's use our best performing model: the GRU-based predictor.
It uses deep [GRU-units](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) that can take into account sequence specific higher-order information derived from training data.
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
data['deep_normalized'] = data.ccs_predicted_deep / np.sqrt(data.mz.values)

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

# extract values to correlate
x = charge_2.deep_normalized.values
y_gravy = charge_2.gravy.values
y_helix = charge_2.helix.values

print('Gravy Pearson:', np.round(pearsonr(x, y_gravy), 2))
print('Helix Pearson:', np.round(pearsonr(x, y_helix), 2))
```
This gives us pearson correlation and p values for both gravy and helicality analysis:

```python
Gravy Pearson: [0.49 0.  ]
Helix Pearson: [0.52 0.  ]
```

Once again let's visualize this to get a better feel for what the numbers are telling us:

```python
from sklearn.linear_model import LinearRegression
from mpl_toolkits.axes_grid1 import make_axes_locatable

def line(x, a, b):
    return x * a + b

reg_gravy = LinearRegression().fit(np.expand_dims(x, -1), np.expand_dims(y_gravy, -1))
reg_helix = LinearRegression().fit(np.expand_dims(x, -1), np.expand_dims(y_helix, -1))

y_line_gravy = [line(x, reg_gravy.coef_, reg_gravy.intercept_) for x in charge_2.deep_normalized.values]
y_line_helix = [line(x, reg_helix.coef_, reg_helix.intercept_) for x in charge_2.deep_normalized.values]

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

As we can observe, our predictor is able to reproduce findings that were already postulated by Chang et al. as well as Meier et al.: Higher gravy and helicality values indeed lead to higher CCS values (at least with respect to our trained predictor). 
This correlation is by no means perfect, but it lies in the nature of complex interactions that lead to a peptide's 3D structure that they cannot easily be modelled by a simple set of descriptors. 
Ultimately, this is why a complex function modelling technique like Deep Learning can add something new!
Implement your own ideas to uncover driving factors like amino acid counts or specific AA positions by altering [this notebook](/notebook/MobilityDrivingFactors.ipynb).

---
### Implementing a custom deep CCS predictor
Say you come up with your very own idea for a deep CCS predictor architecture and want to build on top of ionmob.
It is recomended that you have a NVIDIA CUDA enabled GPU with cuDNN bianries available in your working environment,
otherwise training may take quite some time.
We  will assume that a dataset for training was already generated, including all necesarry steps for preprocessing.
For this demonstration, we can use ionmob datasets. 
Let's use sets from different sources for training, validation and test.
This way, we make sure that we do not overestimate model performace.
We will start our model implementation by fitting a tokenizer.
```python
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from datetime import datetime

import os
# suppress CUDA specific logs 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

from matplotlib import pyplot as plt
from ionmob.alignment.experiment import Experiment

from ionmob.models.deep_models import ProjectToInitialSqrtCCS
from ionmob.preprocess.data import get_tf_dataset
from ionmob.preprocess.helpers import get_sqrt_slopes_and_intercepts, sequence_to_tokens, sequence_with_charge, fit_tokenizer

data_train = pd.read_hdf('data/Meier.h5')
data_valid = pd.read_hdf('data/Tenzer.h5')
data_test = pd.read_hdf('data/Chang.h5')

# tokenize sequences 
seq_tokenized = [sequence_to_tokens(s, drop_ends=True) for s in data_train.sequence.values]
# fit a tokenizer
tokenizer = fit_tokenizer(seq_tokenized)
# have a look at tokens
print(tokenizer.word_index)
```

The tokenizer now knows 41 tokens, 20 of which are Amino Acids and 21 are PTMs.

```python
{'L': 1, 'E': 2, 'S': 3, 'A': 4, 'V': 5, 'D': 6, 'G': 7, 'P': 8, 'T': 9, 'I': 10, 'Q': 11, 'K': 12, 'N': 13, 'R': 14, 'F': 15, 'H': 16, 'Y': 17, 'M-OX': 18, 'C': 19, 'M': 20, 'W': 21, 'A-AC': 22, 'M-OX-AC': 23, 'S-AC': 24, 'M-AC': 25, 'T-AC': 26, 'G-AC': 27, 'V-AC': 28, 'E-AC': 29, 'P-AC': 30, 'C-AC': 31, 'L-AC': 32, 'K-AC': 33, 'D-AC': 34, 'N-AC': 35, 'Q-AC': 36, 'R-AC': 37, 'I-AC': 38, 'F-AC': 39, 'H-AC': 40, 'Y-AC': 41}
```

It has proven to be a very efficient way to build on top of a simple sqrt-fit to help a deep predictor reach high accuracy  as well as fast convergence. 
Ionmob implements its own layer that is able to project all charge states at the same time, making it very convenient to add it to your own predictor.
It is done in two steps: first, fit slopes and intercepts for the initial prediction separately. 
Second, use the gained values to initialize an initial projection layer.
Ionmob makes use of charge state one-hot encoding to gate the prediction based on a given charge state.
If you are interested in the intrinsics, [have a look at the implementation](https://github.com/theGreatHerrLebert/ionmob/blob/8f9378c51149d9e1df89fc4550baeebed2176a22/ionmob/models/deep_models.py#L20).

```python
slopes, intercepts = get_sqrt_slopes_and_intercepts(data_train.mz, data_train.charge, data_train.ccs)
initial_layer = ProjectToInitialSqrtCCS(slopes, intercepts)

# just make sure that everything worked by testing the projection
initial_ccs = initial_layer([np.expand_dims(data_train.mz, 1), tf.one_hot(data_train.charge - 1, 4)]).numpy()

# visualize to make sure all went as intended
plt.figure(figsize=(8, 4), dpi=120)
plt.scatter(data_train.mz, initial_ccs, s=10, label='sqrt projection')
plt.xlabel('Mz')
plt.ylabel('CCS')
plt.legend()
plt.show()
```

<p align="center">
  <img src="docs/images/sqrt_fit.png" width="500" title="sqrt fit">
</p>

The most flexible way to implement a new predictor is to subclass a [tensorflow module or keras model](https://www.tensorflow.org/guide/keras/custom_layers_and_models). 
We will do the latter, as it is the prominent way to generate new predictors for ionmob. 
Let's set up a predictor that uses 1-D convolutions to extract additional information from the sequence of an ion. 
All layers that should be part of the model are defined in the constructor, the execution is defined by specifying the call method.

```python
class ConvolutionalCCSPredictor(tf.keras.models.Model):
    
    def __init__(self, slopes, intercepts, num_tokens=44, seq_len=50):
        super(ConvolutionalCCSPredictor, self).__init__()
        # the inital sqrt projection
        self.initial = ProjectToInitialSqrtCCS(slopes, intercepts)
        
        # the deep sequence processor
        self.embedding = tf.keras.layers.Embedding(input_dim=num_tokens + 1, output_dim=128, input_length=seq_len)
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=8, activation='relu')
        self.mp1d = tf.keras.layers.MaxPool1D(pool_size=2)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu')
        
        # the deep regression tail
        self.dense = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        # read inputs
        mz, charge, sequence, _, _ = inputs
        
        # calculate sequence part
        deep = self.conv1d_2(self.mp1d(self.conv1d(self.embedding(sequence))))
        
        # concat with mz and charge
        concat = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(deep), tf.sqrt(mz), charge])
        
        # deep regression
        dense = self.dense_2(self.dropout(self.dense(concat)))
        
        # output is sqrt-fit + deep-regression
        return self.initial([mz, charge]) + self.out(dense)
```

Callbacks are a convenient way to further automate your training procedure. 
We will use two different callbacks that observe model performance on validation data.
The first one is a learning rate reducer: Should the loss not go down after three consecutive epochs on the validation set, the reducer is going to reduce the learning rate by an order of magnitude.
If there is still no improvement on performance, the early stopper will stop the training procedure after another 2 epochs.

```python
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=1e-1,
    patience=2,
    monde='auto',
    min_delta=1e-5,
    cooldown=0,
    min_lr=1e-7
)

cbs = [early_stopper, reduce_lr]
```

We are now ready to instanciate our predictor, build it and then compile it with a desired objective function and optimizer. 
The models' summary tells us that it has a total of 178,785 trainable parameters.

```python
# create a recurrent predictor
model = ConvolutionalCCSPredictor(slopes, intercepts)

# set input shapes: mz, charge_one_hot, max_seq_len, helix_score, gravy_score
model.build([(None, 1), (None, 4), (None, 50), (None, 1), (None, 1)])

model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
              optimizer=tf.keras.optimizers.Adam(1e-2), metrics=['mae'])

tf_train = get_tf_dataset(data_train.mz, data_train.charge, data_train.sequence, 
                          data_train.ccs, tokenizer, 
                          drop_sequence_ends=True, add_charge=False).shuffle(int(1e7)).batch(1024)

tf_valid = get_tf_dataset(data_valid.mz, data_valid.charge, data_valid.sequence, 
                          data_valid.ccs, tokenizer, 
                          drop_sequence_ends=True, add_charge=False).shuffle(int(1e7)).batch(1024)

tf_test = get_tf_dataset(data_test.mz, data_test.charge, data_test.sequence, 
                          data_test.ccs, tokenizer, drop_sequence_ends=True, add_charge=False).batch(1024)

history = model.fit(tf_train, validation_data=tf_valid, 
                    epochs=50, verbose=False, callbacks=cbs)

# plot training and validation loss 
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
```

<p align="center">
  <img src="docs/images/loss_train_valid.png" width="500" title="training and validation loss">
</p>

As we can see from the plot above, loss quickly stops to improve on validation data while it is still falling on training data. The reduction of the learning rate is clearly visible after epoch 9. We can now have a look at test performance and report our CCS prediction accuracy.

```pytho
model.evaluate(tf_test)

4/4 [==============================] - 0s 16ms/step - loss: 11.5374 - mae: 11.5374

[11.537385940551758, 11.537385940551758]
```
It is arround 11.5. Not too bad compared to the naive approach which gave us a value of arround 13. Want to try it yourself? Use [this notebook](notebook/DeepModelTraining.ipynb).

[^fn1]: Deep learning the collisional cross sections of the peptide universe from a million experimental values. Nat Commun, 2021. https://doi.org/10.1038/s41467-021-21352-8
[^fn2]: Sequence-Specific Model for Predicting Peptide Collision Cross Section Values in Proteomic Ion Mobility Spectrometry. Journal of Proteome Research, 2021. https://doi.org/10.1021/acs.jproteome.1c00185
