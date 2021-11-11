# ionmob
### A Framework for Predicting Collision Cross Section Values of Peptide-Ions with Traditional and Deep Machine Learning Methods

```ionmob``` is a python package for predicting **CCS** values of peptides.
Not only does it contain several pre-trained regression models for this task, but it also introduces a full pipeline to link data preprocessing, model training and CCS value inference.
Models are implemented with recent versions of either [TensorFlow](https://www.tensorflow.org/) or [scikit-learn](https://scikit-learn.org/stable/). 
Please feel free to use, alter or extend ```ionmob``` in any way that suits you as it is free and open source under the **GNU General Public License v3.0**. 
Feel also free to let us know about missing functionality, bugs, or contributions you would like to make!

### What is a peptide CCS value?
The rotationally-averaged collision cross-section - **CCS** - is a scalar value that describes a physical property of an ion with respect to a charge neutral gas. 
It can be directly linked to its ion mobility. 
The ion mobility is used as an additional separating dimension in high throughput mass spectrometry.
It supplements the measurements of retention times and mass-to-charge ratios and ultimately leads to improved peptide identification.

### Why do we measure CCS values of ions?
The CCS value of an ion is a coarse descriptor of its 3D structure. 
Since peptides are chains (strings) of amino acids, there exist permuations in nature that have exactly the same mass and chemical properties. 
Distinguishing between such peptides with conventional methods like e.g. LC-MS-MS is therefore challenging.
Furthermore, post translational modifications (PTMs) might have only a small impact on an ions mass but alter the functionality of a protein.
Since both a permutation of sequnce as well as PTMs have significant impact on 3D structure, one can use ion mobility separation to distinguish between them.
CCS value calculations then give us a measure on how extensively their rotationally-averaged collision cross-section differed.

### Why would I want to predict CCS values of peptides in silico?
First, a predictor might give you insight into factors that drive ion mobility.
This information could then be used to optimize your laboratory workflows or uncover yet unknown relationships.
Second, the high reproducability of measured CCS values in the lab make it an ideal candidate to increase confidence in peptide identifications from database searches.
We think: The recent triumph of ion mobility enhanced mass spectrometry pave the way for expressive predictors by providing previously unavailable ammounts of training data!

### I am NOT a machine learning expert, can I still use ```ionmob```?
Definetly yes! 
We implemented and pretrained models of different complexity that allow for in silico prediction of CCS values for peptide ions of different charge states out-of-the-box. 
They are easily integratable into your existing proteomics workflows.
All you need is a little bit of python scripting experience.
A short introduction can be found down below, or you could have a look at our collection of example notebooks!

### I AM a machine learning expert/experienced coder, what is in it for me?
We made sure that our framework provides a modular set of tools for more experienced coders that want to implement their own models, training strategies or data preprocessing pipelines. 
Have a look at our example notebooks for advanced workflow implementation. 
Feel also free to contribute any optimizations, models or ideas that you come up with.
This will ultimately help to push prediction accuracy to a point where it provides a huge benefit for rescoring of peptide identifications!

### A simple example of inference
Let us assume that you want to have a look at a predictors performance on your own data of peptide identifications that came from some source. 
For ionmob models, you should at least have the following information per peptide: [mz, charge, sequence (CCS)]. 
CCS values are optional but are obviously required if you want to compare CCS predictions to CCS measurements.
A table ready to be used could look like this:
```python

```
This is what the table looks like:

|       mz |   charge | sequence                                    |     ccs | origin     |
|---------:|---------:|:--------------------------------------------|--------:|:-----------|
|  801.89  |        2 | \_AAAAAAAAGGAGDSGDAVTK\_                      | 433.825 | Tenzer-lab |
| 1482.86  |        3 | \_AAAAAPASEDEDDEDDEDDEDDDDDEEDDSEEEAMETTPAK\_ | 701.41  | Tenzer-lab |
|  410.205 |        2 | \_AAAACLDK\_                                  | 296.58  | Tenzer-lab |
|  471.28  |        2 | \_AAAAVVAAAAR\_                               | 348.332 | Tenzer-lab |
|  516.27  |        3 | \_AAADALSDLEIKDSK\_                           | 467.791 | Tenzer-lab |


```python
import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from ionmob.preprocess.data import sqrt_model_dataset

# read data and a predictor
data = pd.read_hdf('test_data.h5')
sqrtModel = tf.keras.models.load_model('pretrained-models/SqrtModel')

# create a dataset for prediction and append predicted ccs values to it
tensorflow_ds = sqrt_model_dataset(data.mz, data.charge, data.ccs).batch(1024)
ccs_predicted = sqrtModel.predict(tensorflow_ds)
data['ccs_predicted'] = ccs_predicted

# plot ground truth vs prediction 
color_dict = {2:'blue', 3:'orange', 4:'lightgreen'}
plt.figure(figsize=(8, 4), dpi=120)
plt.scatter(data.mz, data.ccs, s=10, alpha=.5, label='ground truth')
plt.scatter(data.mz, data.ccs_predicted, s=10, alpha=.5, c='red', label='prediction')
plt.legend()
plt.show()
```

<p align="center">
  <img src="docs/images/sqrt_model.png" width="700" title="prediction vs ground truth">
</p>

```python
sqrtModel.evaluate(tensorflow_ds)
```
4/4 [==============================] - 0s 3ms/step - loss: 2.8155 - mae: 15.1297

[2.815500259399414, 15.129722595214844]
