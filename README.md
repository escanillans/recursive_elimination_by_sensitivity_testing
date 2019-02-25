# recursive_elimination_by_sensitivity_testing
This repo provides a novel feature selection algorithm called Recursive Feature Elimination by Sensitivity Testing (RFEST).

Assumptions: 
1. Binary data (including class label).
2. Data is clean, contains no missing values (e.g. NA, NaN), and is called *final_data.csv*.
3. Class label is called **class_label** and is at the end of your dataset (i.e. it is the last column).
4. Unless altered, this algorithm uses a support vector machine (SVM) with an RBF kernel.

RFEST can be summarized by the following steps:
1. Train an SVM model and output an accuracy measurement, AUC.
2. Compute the importance of each feature (by comparing the decrease in accuracy after perturbing each feature).
3. Remove a percent of those feature(s) based on the computed importance.
4. Repeat until the stopping criterion is met.

To begin, you have the option to run a Jupyter notebook that contains a standard set of machine learning algorithms. This performs cross validation on your entire dataset using the following ML methods: decision tree, random forest, support vector machine (rbf, polynomial, and linear), linear regression, and logistic regression. The notebook is called *baseline_runs*.

Then run *main_program.py*. To test how well RFEST performs on your dataset, we compute a 10-fold cross validation (CV). The results will be saved as a list of AUCs, F1 scores, and ROC curves for each fold. We then split the data into an 80/20-split and perform RFEST again to get a final subset of features. 

Parameters you can adjust:
- **percent** (*main_program.py*) = percent of features you would like to remove at each iteration. Set as a list so that the program will output the results into separate files based on the percent(s) you have chosen. Note that you will need to manually create these folders prior to running or it will not be saved.
- Stopping criteria (*functions.R* and *main_program.py*) = this parameter is set such that the algorithm stops when we get a 5% drop (search for **95** or **0.95** in algorithm) in AUC (i.e. in any particular iteration an AUC is computed and compared to the max AUC achieved thus far).
- **n_splits** (*roc_curves.ipynb*) = the number of folds for stratified CV.

What is outputted in *results/xPercentRemoval* folder (where *x* is percent you chose) when you run *main_program.py*:
1. 10-fold CV.
2. List of relevant features.

What is outputted when you run *roc_curves.ipynb*:
1. AUC for each model (from 10-fold CV).
2. ROC curves for each model (also showing the ROC curve per fold).
3. F1-Score for each model (from 10-fold CV).

Data Provided:
I have provided a dataset based on the Correlation Immune (CI) function of order two, known as the parity function. The dataset contains 50 features and 1000 instances. Two randomly chosen features were used to create the class label. 

The goal is to see whether RFEST can detect what those features are. Visit the link at the bottom to read more on CI functions.

Notes:
- No extra tuning is done before the algorithm runs. Default parameters for the machine learning algorithms have been used. This can be changed accordingly.
- RFEST is a generalized feature selection algorithm. Meaning that one can, in practice, use any machine learning algorithm for evaluating feature relevance (refer to paper for more details). 
- Before selecting your choice of machine learning algorithm to use with RFEST, it would be wise to run a set of general machine learning methods such as the following: decision tree, random forest, SVM (RBF, polynomial, and linear kernels), linear regression, logistic regression. 
- Note that some of the processes were written in R, but the bulk of the algorithm is written in Python. This should not be a big concern.

Post-hoc Analysis:
If you are interested in pairwise interactions, I've uploaded a script called *LinearSVMFinalAnalysis.R*. Using a linear SVM, this will compute a ranking of the individual features and their interaction pairs. 

Note that you will probably need to change the variables **listOfFeatures** and **file** in your implementation of *LinearSVMFinalAnalysis.R*. 

To read more on RFEST, visit this [link](https://ieeexplore.ieee.org/document/8614039). 





