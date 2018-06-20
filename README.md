# recursive_elimination_by_sensitivity_testing
This repo provides a novel feature selection algorithm called Recursive Feature Elimination by Sensitivity Testing.

Assumptions: 
1. Binary data (including class label).
2. Data is clean, contains no missing values (e.g. NA, NaN), and is called *final_data.csv*.
3. Class label is called **class_label** and is at the end of your dataset (i.e. it is the last column).
4. Unless altered, this algorithm uses an support vector machine (SVM) with an RBF kernel.

The process of this recursive algorithm can be summarized by the following steps:
1. Train an SVM model and output an accuracy measurement, AUC.
2. Compute the importance of each feature (by comparing the decrease in accuracy after perturbing each feature).
3. Remove a percent of those feature(s) based on the computed importance.
4. Repeat until the stopping criterion is met.

To begin the algorithm, run *MainProgram.R* [source("MainProgram.R")] There are two main processes involved with *MainProgram.R*:
1. 10-fold cross validation (CV) is used for RFEST to return the efficacy of the model (in terms of area under the ROC curve, AUC).
2. In practice, to actually obtain the relevant subset of features, we do an 80-20 split of the whole dataset and run RFEST. 

Parameters you can adjust:
- **numFolds** = number of folds for stratified CV.
- **percents** = percent of features you would like to remove at each iteration. Set as a list so that the program will output the results into separate files based on the percent(s) you have chosen. Note that you will need to manually create these folders prior to running or it will not be saved.
- **seeds** = random seeds. You may select more than one if you would like to see if there is a big discrepancy in results.
- Stopping criteria = this parameter is set such that the algorithm stops when we get a 5% drop (search for **95** in algorithm) in AUC (i.e. in any particular iteration an AUC is computed and compared to the max AUC achieved thus far). You can find this parameter in functions.R.
- **numCores** = number of cores you would like to use in parallel (e.g. for my case, I used 10-fold CV and so numCores was set to 10 to run in parallel by the number of folds). This variable can be found in the Main() function in functions.R.

What is outputted in *results/xPercentRemoval* folder (where *x* is percent you chose):
1. Final AUC for each model (from 10-fold CV).
2. Highest AUC achieved for each model (before reaching stopping criterion) (from 10-fold CV).
3. Runtime for each model (from 10-fold CV).
4. An average AUC and average runtime (from 10-fold CV).
5. List of features returned.
6. AUC before breaking and runtime when performing RFEST on 80-20 split. 

Notes:
- No extra tuning is done before the algorithm runs. Default parameters for the machine learning algorithms have been used. This can be changed accordingly.
- RFEST is a generalized feature selection algorithm. Meaning that one can, in practice, use any machine learning algorithm for evaluating feature relevance (refer to paper for more details). 
- Before selecting your machine learning algorithm, it would be wise to run a set of general machine learning methods such as the following: decision tree, random forest, SVM (RBF, polynomial, and linear kernels), linear regression, logistic regression. 
- Recall that RFEST was written with the programming language R and the baseline methods were implemented in Python. This should not be a big concern. However, to be certain that it is not, you can perform CV in Python. Specifically, RFEST will output a list of features. Take only those features and run an SVM with an RBF kernel (or whatever method you chose to run in RFEST) in Python and the results should be similar.

Post-hoc Analysis:
If you are interested in pairwise interactions, I've uploaded a script called *LinearSVMFinalAnalysis.R*. Using a linear SVM, this will compute a ranking of the individual features and their interaction pairs. 

Note that you will probably need to change the variables **listOfFeatures** and **file** in your implementation. 


To read more on RFEST, visit this [link](https://escanillans.github.io/ResearchPapers/rfest.pdf).





