# recursive_elimination_by_sensitivity_testing
This repo provides a novel feature selection algorithm called Recursive Feature Elimination by Sensitivity Testing (RFEST).

Assumptions: 
1. Binary data (including class label).
2. Data is clean, contains no missing values (e.g. NA, NaN), and is called *final_data.csv*.
3. Class label is called **class_label** and is at the end of your dataset (i.e. it is the last column).
4. Unless altered, this algorithm uses an support vector machine (SVM) with an RBF kernel.

RFEST can be summarized by the following steps:
1. Train an SVM model and output an accuracy measurement, AUC.
2. Compute the importance of each feature (by comparing the decrease in accuracy after perturbing each feature).
3. Remove a percent of those feature(s) based on the computed importance.
4. Repeat until the stopping criterion is met.

To begin, run *MainProgram.R* [source("MainProgram.R")]. To test how well the algorithm performed, we will compute a 10-fold cross validation (CV) using only the features returned by RFEST. Open python notebook *roc_curves.ipynb*. This notebook will perform 10-fold CV on two different algorithms:
1. Linear SVM built using your whole dataset.
2. Non-linear SVM (default is an RBF kernel) built using the subset of features returned by RFEST.

Parameters you can adjust:
- **percents** (*MainProgram.R*) = percent of features you would like to remove at each iteration. Set as a list so that the program will output the results into separate files based on the percent(s) you have chosen. Note that you will need to manually create these folders prior to running or it will not be saved.
- **seeds** (*MainProgram.R*) = random seeds. You may select more than one if you would like to see if there is a big discrepancy in results.
- Stopping criteria (*functions.R*) = this parameter is set such that the algorithm stops when we get a 5% drop (search for **95** in algorithm) in AUC (i.e. in any particular iteration an AUC is computed and compared to the max AUC achieved thus far).
- **n_splits** (*roc_curves.ipynb*) = the number of folds for stratified CV.

What is outputted in *results/xPercentRemoval* folder (where *x* is percent you chose) when you run *MainProgram.R*:
1. AUC before breaking and runtime when performing RFEST on 80-20 split.
2. List of features.

What is outputted when you run *roc_curves.ipynb*:
1. AUC for each model (from 10-fold CV).
2. ROC curves for each model (also showing the ROC curve per fold).
3. F1-Score for each model (from 10-fold CV).

Data Provided:
I have provided a dataset based on the Correlation Immune (CI) function of order two, known as the parity function. Visit the link at the bottom to read more on CI functions.

The dataset contains 50 features and 1000 instances. Two randomly chosen features were used to create the class label. The goal is to see whether RFEST can detect what those features are. The results in the Python notebook shows the features and the CV results.

Notes:
- No extra tuning is done before the algorithm runs. Default parameters for the machine learning algorithms have been used. This can be changed accordingly.
- RFEST is a generalized feature selection algorithm. Meaning that one can, in practice, use any machine learning algorithm for evaluating feature relevance (refer to paper for more details). 
- Before selecting your machine learning algorithm, it would be wise to run a set of general machine learning methods such as the following: decision tree, random forest, SVM (RBF, polynomial, and linear kernels), linear regression, logistic regression. 
- Recall that RFEST was written with the programming language R and the baseline methods and final CV for RFEST were implemented in Python. This should not be a big concern.

Post-hoc Analysis:
If you are interested in pairwise interactions, I've uploaded a script called *LinearSVMFinalAnalysis.R*. Using a linear SVM, this will compute a ranking of the individual features and their interaction pairs. 

Note that you will probably need to change the variables **listOfFeatures** and **file** in your implementation. 

To read more on RFEST, visit this [link](https://escanillans.github.io/ResearchPapers/rfest.pdf).





