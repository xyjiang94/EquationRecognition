# EquationRecognition

## Author
- [Jiadong Yan](https://github.com/FrankYan93)  
- [Xinyi Jiang](https://github.com/xyjiang94)  
- [Zhengyang Zhou](https://github.com/zhengyjo)

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
3. `python predict.py [img folder path]`

## Some Tips and Interesting findings
1. Binary or Gray Scale; Segmentation VS training set (the "+")
2. Picture deformation
3. Drop out Rate
4. Appropriate attributes/features to represent the labels
5. Whether to classify the equations
6. Batch size adjustment
7. TBD
8. Input size adjustments
9. Deal with ".", "-", "x"

## Version
0.1
