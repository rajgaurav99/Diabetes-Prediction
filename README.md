# Diabetes-Prediction
Diabetes is one of the deadliest diseases in the world. It is not only a disease but also creator of different kinds of diseases like heart attack, blindness etc. The normal identifying process is that patients need to visit a diagnostic center, consult their doctor, and sit tight for a day or more to get their reports.
Problem Statement 1: Classify the patients into diabetic or non-diabetic
Problem Statement 2: Cluster diabetic patients into severe or not severe.


To predict the outcome: Run GUI.py

To view Cluster Information: Run Cluster_Formation.py

NOTE: Be sure you have sklearn and Tkinter libraries installed on your system for Python. If not then install them first along with other libraries required given below. Spyder IDE is recommended.


Design and Implementation

Prediction/Classification
A model is made by combining 3 classification algorithms. Predicted outcome is taken from each classifier and the outcome in the best of three is taken as the final outcome like a voting mechanism. The 3 algorithms used are:
•	ID3 Decision Tree: In decision tree learning, ID3 (Iterative Dichotomiser 3) is an algorithm invented by Ross Quinlan used to generate a decision tree from a dataset. ID3 is the precursor to the C4.5 algorithm, and is typically used in the machine learning and natural language processing domains. The ID3 algorithm is used by training on a data set  to produce a decision tree which is stored in memory. At runtime, this decision tree is used to classify new test cases (feature vectors) by traversing the decision tree using the features of the datum to arrive at a leaf node. The class of this terminal node is the class the test case is classified as.

•	Naïve Bayes Classifier: In machine learning, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naive) independence assumptions between the features. Naive Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression: which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.
•	KNN Classifier: In pattern recognition, the k-nearest neighbours algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression. In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbours, with the object being assigned to the class most common among its k nearest neighbours (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbour.

Clustering
After the Classification is done and the outcome is Diabetes positive, then the subject is clustered into one of the 2 clusters:
	1. Diabetes not severe at the moment
	2. Diabetes severe and immediate medical help required
The clustering is only done by taking the attributes Glucose Level and insulin level into consideration since they are the most vital attributes for a diabetes patient.
•	K-Means Clustering: K-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. This results in a partitioning of the data space into cells. The problem is computationally difficult (NP-hard); however, efficient heuristic algorithms converge quickly to a local optimum. These are usually similar to the expectation-maximization algorithm for mixtures of Gaussian distributions via an iterative refinement approach employed by both k-means and Gaussian mixture modelling. They both use cluster centers to model the data; however, k-means clustering tends to find clusters of comparable spatial extent, while the expectation-maximization mechanism allows clusters to have different shapes.

GUI Interface
Rather than using a simple CUI program, a GUI interface has been used for collecting inputs and well as displaying the output. GUI interface for python can be implemented using tKinter. It is a standard Python interface to the Tk GUI toolkit shipped with Python. Python with tkinter outputs the fastest and easiest way to create the GUI applications. 

Tools and Libraries Used Used
Spyder
	Spyder is a powerful scientific environment written in Python, for Python, and designed by and for scientists, engineers and data analysts. It offers a unique combination of the advanced editing, analysis, debugging, and profiling functionality of a comprehensive development tool with the data exploration, interactive execution, deep inspection, and beautiful visualization capabilities of a scientific package.

Sklearn
  Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
Functions Used:
sklearn.cluster.KMeans(): Clustering using Kmeans method.
tree.DecisionTreeClassifier(): Classifying using decision tree method.
sklearn.neighbors.KNeighborsClassifier(): Classifying using knn method.
sklearn.naive_bayes.GaussianNB(): Classifying using Gaussian naïve bayes method.

