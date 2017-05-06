# Report of EquationRecognition

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
`python predict.py [img folder path]`

## Build Instructions
1. `python preprocessing.py` to generate shelf of training data.(You need annotated folder at this directory.)  
2. `python MER_NN.py train [the address you want to save your model]`

## Introduction of the frame of project
The project has two main part. The first part is the neural networks that is trained to recognize isolated symbols. The other part is the algorithm to partition the image correctly, as long as recognize the equation. The second part is inspired by the paper written by NE Matsakis.[1]  

At the very first beginning, we decided to use convolutional neural network to recognize the equations directly without segmentation, but we failed as we saw the model could cause more confusions when predicting. Since the time is limited, we swtich our way to deal with this project in the current way.

For the neural network, we construct the training model based on Zhihao's CNN on digit-recognizaiton. In order to make the model more efficient, we apply a lot of techniques such as size wrapping, deformation, drop-out rate adjustment and so on, which we will elaborate in **Some Tips and Interesting findings**. Generally, we find that in order to recognize **38** different symbols, we need to include more features on the input of the readout layer. In addition, in case the F(X) is too small, we also have some skip layers to mitigate the effect of vanishing gradient for bottom layer like the following:

![google]
(https://ooo.0o0.ooo/2017/05/07/590e299547c07.png)  

Inspired by the Deep Residual Neural Network, we come up with the following structure with 1024 features in the final input of readout layers (input and output are recorded as number of features): 

1. First layer of convolution: 3 X 3 window, input:1, output:32  

2. a. Second layer of convolution: 3 X 3 window, input:32,output:32  
   b. Skip layer: 3 X 3 window, input:32,output:64

3. Third layer of convolution: 3 X 3 window, input:32, output: 64

4. a. Forth layer of convolution: 3 X 3 window, input:64,output:64  
   b. Skip layer: 3 X 3 window, input:64,output:128  

5. Fifth layer of convolution: 3 X 3 window, input:64, output: 128  

6. Sixth layer of convolution: 3 X 3 window, input:128, output: 128
7. Densly Connected layer: 1 X 1 window, input:128, output: 1024  
8. readout layer: input:1024, output: 38



In term of part two, the partition algorithm, the basic idea is to divide the equation into different strokes, and find the best way to partition the strokes into symbols. There are 4 steps:

1. divide the equation into different strokes using connected components algorithm
There's an hidden assumption that each stroke belongs to a single symbol, and only stroke from one symbol could be connected.

2. establish a minimum spanning tree (MST) of strokes
Assume that we have n isolated strokes, then the possible ways to partition those strokes will be a big number. Let f(N) be the the number of stroke sets that are examined as possible symbols, then f(N) = 2^N. N is the number of strokes.
To reducing this problem to a manageable size distance, we establish a minimum spanning tree (MST), and only consider the partition of its subtree. The vertices of MST are the strokes, the weight of the edges between two strokes are the Euclidean distance between the centroids of their bounding boxes. Therefore, only strokes that are near each other will be combined to test if it will be a valid symbol. This reduce the complexity to f(M) = 2^M, where M is the number of edges.

3. find the best partition of strokes
The basic idea is to try different way of partition the strokes, in other words, different ways of combining strokes, to find the best one. Every combined stroke will be recognized by the trained neural networks and will return a list of likelihood. The likelihood of a partition is the sum of the likelihood of every symbol that belongs to it. The best partition will have the highest overall likelihood.

However, this is still a complex algorithm, we simplify it further, which will be discussed in detail in introduction of `partition.py`

4. classify the image according to the recognized symbols
Step 3 gives us a list of symbols and their bounding box. We can use a simple classification algorithm to map the list to one of the 35 equations. Read introduction of `classifyEq.py` for more details



## Some Tips and Interesting findings
1. Binary or Gray Scale; Segmentation VS training set  
	* The first condition you have to meet is that you should use the same/unified format in both the training set and the segmentation. For example, if you use gray scale in training set, then you should also use gray scale when processing the equations pictures that you want to predict.  
	* After a series of comparisons, we find the gray scale can bring us higher accuracy. So in our training and processing, we use gray scale, instead of the binary mode.
2. Picture deformation
	* Sometimes people may write a symbol in a sloppy way. Then it will be useful to apply the techniques of picture deformation to do the adjustments, such as affine transformation.

3. Drop out Rate  
	* In our model, the drop out rate is 0.5, which means in every step we training, we only randomly keep 50% of our batch size to do the learning. This is to mitigate the overfiting.

4. Appropriate attributes/features to represent the labels  
	* At the very beginning, we had only 64 features for the input of our readout layer and we found that 64 was not enough to represent 38 target symbols. After the symbol complexity analysis and testing, we found 1024 was appropriate.
5. How to classify the equations in the extra part
	* If you want to solve the classification of equations, you should apply a neural network on the equations directly, or make a function to output symbols in some order based on the locations of symbols
6. Batch size adjustment  
	* Originally the batch size was 50, but the efficiency was low. Then after some testing, we found 100 was much better since it could cover more cases in the training.

7. Input picture size adjustments
	* Similary with the gray scale problems, we have to keep the size of the training pictures and the segmented pictures in the same size. If not, the model will not match with the semented pictures, which will cause prediction errors.  

8. The range of each element of picture numpy array
	* Similary with previous one, we have to keep the range of the elements in numpy arrays that the training pictures are transformed to, and the of the segmented pictures the same. For example, if each element in a numpy array of training picture is in [0,1], then the element in the array of segmented pictres should also be in [0,1]. Otherwise, you will have an extremly low accuracy.
8. Deal with ".", "-", "x"

## Python Files's Details

### MER.py
	1. The training model file, which contains the whole structure of the neural network. The input of this one is from DataWrapperFianl.py

### DataWrapperFinal.py
	1. Take the numpy arrays of training pictures and the corresponding labels as input to produce training set
	2. Get_valid method is to use the first 500 records as cross validation sets  
	3. Next_batch method is to produce the batch of training set in each step
### readDB.py
	1. Read the training shelf from preprocessing  
	2. Produce the numpy arrays of training pictures and the corresponding labels. Then we can feed them into DataWrapperFinal.py
### preprocessing.py

### partition.py

### recognize.py

### recognizeFromShelf.py

### predict.py

##Result Analysis

## Reference
1. Matsakis, Nicholas E. Recognition of handwritten mathematical expressions. Diss. Massachusetts Institute of Technology, 1999.
