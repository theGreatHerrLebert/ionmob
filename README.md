# ionmob
A Prediction Framework for Peptide Ion-Mobilities

```ionmob``` is a python package for predicting **CCS** values of peptides.
Not only does it contain several pre-trained regression models for this task, but it also introduces a full pipeline to link data preprocessing, model training and CCS value inference.
Models are implemented with recent versions of either [TensorFlow](https://www.tensorflow.org/) or [scikit-learn](https://scikit-learn.org/stable/). 
Please feel free to use, alter or extend ```ionmob``` in any way that suits you as it free and open source under the **GNU General Public License v3.0**.

### What is a peptide CCS value?
The rotationally-averaged collision cross-section - **CCS** - is a scalar value that describes a physical property of a specific peptide, namely its ion mobility. 
The ion mobility is used as an additional separating dimension in high throughput mass spectrometry.
It supplements the measurements of the retention time and the mass-to-charge ratio and ultimately leads to improved peptide identification.

### Why would I want to predict CCS values of peptides in silico?
The CCS of a peptide is a relatively coarse descriptor of its 3D structure. 
Since peptides with differing mass 
Recently, there has been renewed interest in possible implementations of such a predictor.

### I am NOT a machine learning expert, can I still use ```ionmob```?
Definetly yes! We implemented and pretrained models of different complexity that allow for in silico prediction of peptides out-of-the-box. 
They are easily integratable into your existing proteomics workflows.
All you need is a little bit of python scripting experience.

### I AM a machine learning expert/experienced coder, what is in it for me?
We made sure that our framework provides a modular set of tools for more experienced coders that want to implement their own models, training strategies or data preprocessing pipelines.
