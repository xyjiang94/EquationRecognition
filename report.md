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

## introduction of the frame of project
The project has two main part. The first part is the neural networks that is trained to recognize isolated symbols. The other part is the algorithm to partition the image correctly, as long as recognize the equation. The second part is inspired by the paper written by NE Matsakis.[1]

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
1. Binary or Gray Scale; Segmentation VS training set (the "+")
2. Picture deformation
3. Drop out Rate
4. Appropriate attributes/features to represent the labels
5. Whether to classify the equations
6. Batch size adjustment
7. TBD
8. Input size adjustments
9. Deal with ".", "-", "x"

## Python Files's Details

### preprocessing.py

### partition.py

### recognize.py

### recognizeFromShelf.py

### predict.py

## Reference
1. Matsakis, Nicholas E. Recognition of handwritten mathematical expressions. Diss. Massachusetts Institute of Technology, 1999.
