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

### I AM a machine learning expert/experienced coder, what is in it for me?
We made sure that our framework provides a modular set of tools for more experienced coders that want to implement their own models, training strategies or data preprocessing pipelines.

### A simple example of inference
```python
import pandas as pd
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from ionmob.preprocess.data import sqrt_model_dataset
```

