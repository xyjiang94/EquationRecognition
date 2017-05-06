# EquationRecognition

## Author
- Jiadong Yan  
- Xinyi Jiang  
- Zhengyang Zhou

## Description
Recognize hand-written equations.

## Dependencies
- tensorflow
- scipy
- skimage
- shelve
- imghdr
- PIL
- numpy

## Run Instructions

## Build Instructions
1. `python preprocessing.py` to generate shelf of training data.(You need annotated folder at this directory.)  
2. `python MER_NN.py train [the address you want to save your model]`
3.
## Some Tips and Interesting findings
1.Binary or Gray Scale; Segmentation VS training set (the "+")
2.Picture deformation
3.Drop out Rate
4.Appropriate attributes/features to represent the labels
5.Whether to classify the equations
6.Batch size adjustment
7.TBD
8.Input size adjustments

## Version
0.1
