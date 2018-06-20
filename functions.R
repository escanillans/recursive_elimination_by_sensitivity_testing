# This function creates numFolds folds for use in
# cross validation with stratified sampling.
createFolds <- function(data, numFolds, seed)
{
  set.seed(seed)
  # determine the number of class label values
  classIndex = ncol(data)
  classLabels = data[,classIndex]
  levels(classLabels) = 1:0
  data[,classIndex] = classLabels
  
  # create a list of folds
  folds = vector("list", numFolds)
  
  # apply stratified sampling:
  # separate class labels
  data0 = subset(data, data[,classIndex] == 0, drop = FALSE)
  data1 = subset(data, data[,classIndex] == 1, drop = FALSE)
  
  # record number of rows in each dataset
  instanceIndexData0 = as.numeric(rownames(data0))
  instanceIndexData1 = as.numeric(rownames(data1))
  
  # for each dataset do:
  # pick a random set of instance indices from instanceIndexData0
  sequenceOfInstanceData0 = sample(instanceIndexData0, length(instanceIndexData0), replace = FALSE)
  
  # place random list of numbers in ten folds in sequential order
  j = 1
  while(length(sequenceOfInstanceData0) > 0)
  {
    # if the index number for folds exceeds the number of folds,
    # then set back to first fold
    if(j > length(folds))
    {
      j = 1;
      j = as.numeric(j);
    }
    
    # add the ith element in sequenceOfInstanceData0 to the jth fold
    folds[[j]] = c(folds[[j]], sequenceOfInstanceData0[1])
    
    # increment jth index number
    j = as.numeric(j);
    j = j + 1;
    
    # remove the ith element in sequenceOfInstanceData0
    sequenceOfInstanceData0 = sequenceOfInstanceData0[-1]
  }
  
  # pick a random set of instance indices from instanceIndexData1
  sequenceOfInstanceData1 = sample(instanceIndexData1, length(instanceIndexData1), replace = FALSE)
  
  # place random list of numbers in ten folds in sequential order
  # note: work our way backwards to ensure that each fold has roughly the same size
  j = numFolds
  while(length(sequenceOfInstanceData1) > 0)
  {
    # if the index number for folds exceeds the number of folds,
    # then set back to first fold
    if(j < 1)
    {
      j = numFolds;
      j = as.numeric(j);
    }
    
    # add the 1st element in sequenceOfInstanceData1 to the jth fold
    folds[[j]] = c(folds[[j]], sequenceOfInstanceData1[1])
    
    # decrement jth index number
    j = as.numeric(j);
    j = j - 1;
    
    # remove the 1st element in sequenceOfInstanceData1
    sequenceOfInstanceData1 = sequenceOfInstanceData1[-1]
  }
  
  return(folds)
}

# create a function to change class labels into binary (0,1) encoding
binaryData <- function(data){
  # record class label
  # Note: class label is typically placed in last column of dataset
  classLabelIndex = ncol(data)
  
  # get a vector of the class labels for each instance
  classLabelForEachInstance = data[,classLabelIndex]
  
  # convert to binary
  levels(classLabelForEachInstance) = 0:1
  
  # change dataset to have binary encoding
  data[,classLabelIndex] = classLabelForEachInstance
  
  return(data)
}

# RFEST Algorithm
# Input: data, percent removal, folds
# Output: AUC, RunTime
RFEST <- function(dataSet, percentRemoval, folds, seed, foldNumber)
{
  runtime = proc.time()
  set.seed(seed)
  cat("Currently working on fold number: ", foldNumber, "\n")
  
  # get the appropriate training and testing data
  traindata_whole = dataSet[-folds[[foldNumber]],]
  testdata = dataSet[folds[[foldNumber]],]
  
  # partition the train set into train and tune
  sampleSize = floor(0.8 * nrow(traindata_whole))
  trainIndices = sample(seq_len(nrow(traindata_whole)), size = sampleSize)
  traindata = traindata_whole[trainIndices,]
  tunedata = traindata_whole[-trainIndices,]	
  
  # initialize previous AUC to 0
  bestAUC = 0
  maxAUC = 0
  
  repeat
  { 	      	
    # train model
    # note: C = 1, gamma = 1/num_features
    svmModel <- ksvm(class_label~., data = traindata, 
                     kernel = "rbfdot", type = "C-svc", C = 1, 
                     gamma = 1/(ncol(traindata)-1), prob.model = TRUE, 
                     scaled = append(rep(TRUE, ncol(traindata)-1), FALSE))
    
    
    # predict probabilities on tuning set.
    p <- kernlab::predict(svmModel, tunedata[,-ncol(tunedata)], type = "probabilities")
    
    pred <- prediction(predictions = as.matrix(p[,2]),
                       labels = as.numeric(as.matrix(tunedata[,ncol(tunedata)])))
    
    # calculate AUC on tuning set.
    perf.auc <- performance(pred, measure = "auc")
    AUC <- unlist(perf.auc@y.values)
    cat("fold: ", foldNumber, "current auc: ", AUC, "\n")
    
    # stopping criteria: stop when we get a 5% drop in AUC
    if(AUC <= (0.95*maxAUC))
    {
      # test bestModel on testset
      # build model using only remaining features 
      finalModel <- ksvm(class_label~., data = traindata_whole, 
                         kernel = "rbfdot", type = "C-svc", C = 1, 
                         gamma = 1/(ncol(traindata_whole)-1), prob.model = TRUE, scaled = FALSE)
      
      # predict probabilities on training set (can also just use a separate tuning set).
      finalP <- kernlab::predict(finalModel, testdata, type = "probabilities")
      
      finalPred <- prediction(predictions = as.matrix(finalP[,2]),
                              labels = as.numeric(as.matrix(testdata$class_label)))
      
      # calculate AUC on training set.
      final.perf.auc <- performance(finalPred, measure = "auc")
      finalAUC <- unlist(final.perf.auc@y.values)
      
      # stop timer
      runResults = proc.time() - runtime
      
      # return the model to use in calculating final AUC and return feature list
      model = list(AUC = finalAUC, runResults = runResults, auc_before_break = bestAUC);
      
      return(model) 
      
    }
    else
    {
      # record the current best AUC and the features that were used
      bestAUC = AUC
      if(bestAUC >= maxAUC)
      {
        maxAUC = bestAUC
      }
    }
    
    # implement RFEST
    # create an array to store the list of differences in AUC
    # for each feauture
    aucForEachFeature = c()
    # iterate through each feature
    for(j in 1:(ncol(tunedata)-1))
    {
     
      # save copy of original feature vector
      original = tunedata[,j]
      
      # if continuous, permute. otherwise, flip
      if(length(unique(tunedata[,j])) > 2){
        tunedata[,j] = sample(tunedata[,j], nrow(tunedata), replace = FALSE, 
                              prob = rep(1/length(tunedata[,j]), nrow(tunedata)))
      } else{
        uniqueVals = unique(tunedata[,j])
        
        # if uniqueVals only has one value (due to near zero variance), replace with opposite number
        if(length(uniqueVals) == 1){
          tunedata[,j] = ifelse(tunedata[,j] == 0, 1, 0)
        } else{
          tunedata[,j] = ifelse(tunedata[,j] == uniqueVals[1], uniqueVals[2], uniqueVals[1])
        }
      }
      
      # calculate new probabilities on testing set.
      # note: this will use the model we trained on tunedata
      p <- kernlab::predict(svmModel, tunedata, type = "probabilities")
      
      pred <- prediction(predictions = as.matrix(p[,2]),
                         labels = as.numeric(as.matrix(tunedata[,ncol(tunedata)])))
      
      # calculate AUC on tuning set.
      perf.auc <- performance(pred, measure = "auc")
      AUCflipped <- unlist(perf.auc@y.values)
      
      # take difference between AUC_flipped and AUC_original
      # cote: the more negative, the more relevant
      diffAUC <- AUCflipped - AUC
      
      # add diffAUC to aucForEachFeature
      aucForEachFeature = c(aucForEachFeature, diffAUC)
      
      # flip the value of the jth feature vector
      # in testdata back to its original value
      tunedata[,j] = original
      
      # repeat for the next feature.
    }
    
    # get class label vector for each dataset
    testdataClassLabel = testdata[,ncol(testdata)]
    traindataClassLabel = traindata[,ncol(traindata)]
    traindata_wholeClassLabel = traindata_whole[,ncol(traindata_whole)]
    tunedataClassLabel = tunedata[,ncol(tunedata)]
    
    # create temporary datasets for ordering by diffAUC
    # these datasets do not contain the class label
    testdataTemp = testdata[,-ncol(testdata)]
    traindataTemp = traindata[,-ncol(traindata)]
    traindata_wholeTemp = traindata_whole[,-ncol(traindata_whole)]
    tunedataTemp = tunedata[,-ncol(tunedata)]
    
    # combine temporary datasets with aucForEachFeature
    testdataTemp = rbind(testdataTemp, aucForEachFeature)
    traindataTemp = rbind(traindataTemp, aucForEachFeature)  
    traindata_wholeTemp = rbind(traindata_wholeTemp, aucForEachFeature)
    tunedataTemp = rbind(tunedataTemp, aucForEachFeature) 	
    
    # order testdata and traindata in descending order.
    testdataTemp = testdataTemp[,order(-testdataTemp[nrow(testdataTemp),])]
    traindataTemp = traindataTemp[,order(-traindataTemp[nrow(traindataTemp),])]
    traindata_wholeTemp = traindata_wholeTemp[,order(-traindata_wholeTemp[nrow(traindata_wholeTemp),])]
    tunedataTemp = tunedataTemp[,order(-tunedataTemp[nrow(tunedataTemp),])]
    
    # remove given percent of features
    # record the number of items to delete
    numFeaturesToDelete = ceiling((percentRemoval/100)*(ncol(traindata)-1))
    
    # remove the least relevant feature(s) from datasets
    testdataTemp[1:numFeaturesToDelete] = list(NULL)
    traindataTemp[1:numFeaturesToDelete] = list(NULL)
    traindata_wholeTemp[1:numFeaturesToDelete] = list(NULL)
    tunedataTemp[1:numFeaturesToDelete] = list(NULL)
    
    # if there is 1 variable left, R will create a vector of the temporary data
    # prevent this from happening...
    if(length(colnames(traindataTemp)) == 1)
    {
      # get a copy of the variable name
      varName = colnames(traindataTemp)
      
      # remove the last row from datasets
      testdataTemp = testdataTemp[-nrow(testdataTemp),]
      traindataTemp = traindataTemp[-nrow(traindataTemp),]
      traindata_wholeTemp = traindata_wholeTemp[-nrow(traindata_wholeTemp),]
      tunedataTemp = tunedataTemp[-nrow(tunedataTemp),]
      
      # change back to dataframe and change the name of the column to varName
      testdataTemp = data.frame(testdataTemp)
      traindataTemp = data.frame(traindataTemp)
      traindata_wholeTemp = data.frame(traindata_wholeTemp)
      tunedataTemp = data.frame(tunedataTemp)
      
      colnames(testdataTemp) = varName
      colnames(traindataTemp) = varName
      colnames(traindata_wholeTemp) = varName
      colnames(tunedataTemp) = varName
    }
    else
    {
      # remove the last row from datasets
      testdataTemp = testdataTemp[-nrow(testdataTemp),]
      traindataTemp = traindataTemp[-nrow(traindataTemp),]
      traindata_wholeTemp = traindata_wholeTemp[-nrow(traindata_wholeTemp),]
      tunedataTemp = tunedataTemp[-nrow(tunedataTemp),] 
    }
    # make sure the class labels are factored
    testdata = cbind(testdataTemp, class_label = testdataClassLabel)
    traindata = cbind(traindataTemp, class_label = traindataClassLabel)
    traindata_whole = cbind(traindata_wholeTemp, class_label = traindata_wholeClassLabel)
    tunedata = cbind(tunedataTemp, class_label = tunedataClassLabel)
  }
}

# Main implementation is the function that you will run
# to compute the AUC and run time for 10 fold cv
Main <- function(dataSet, numFolds, percentRemoval, algorithm, seed)
{

  # make class label values binary
  dataSet = binaryData(dataSet)
  
  # create k folds, where k = numFolds
  folds = createFolds(dataSet, numFolds, seed)
  
  # create a list of AUCs, a list of features, and a list of runtimes for each fold
  aucForEachModel = c()
  runtimesForEachModel = c()
  auc_before_breakForEachModel = c()
  
  # run k fold in parallel
  numCores = 10

  # create a vector representing the fold numbers
  values = 1:numFolds
  
  # apply parallel processing for the number of outerfolds
  cat("running ", algorithm, "\n")
  
  kFoldCvResults = mclapply(values, RFEST, dataSet = dataSet, percentRemoval = percentRemoval, 
                            folds = folds, seed = seed, mc.cores = numCores)
  
  for(i in 1:length(kFoldCvResults))
  {
    # add model values to corresponding lists
    aucForEachModel = c(aucForEachModel, kFoldCvResults[[i]]$AUC)
    
    # note that for run time, we will collect the total user 
    # time run (third column of run time results)
    runtimesForEachModel = c(runtimesForEachModel, kFoldCvResults[[i]]$runResults[3])
    
    auc_before_breakForEachModel = c(auc_before_breakForEachModel, kFoldCvResults[[i]]$auc_before_break)
  }
  
  # create final results list for k fold cv 
  result = list(AUCs = aucForEachModel, RunTimes = runtimesForEachModel, AUCs_before_break = auc_before_breakForEachModel)
  
  return(result)
  
}

# RFEST Algorithm
# Input:
# Output: Final feature subset, run time
RFEST_Features <- function(dataSet, percentRemoval, folds, foldNumber)
{
  runtime = proc.time()

  # get the appropriate training and testing data
  traindata = dataSet[-folds[[foldNumber]],]
  tunedata = dataSet[folds[[foldNumber]],]
  
  # initialize previous AUC to 0
  bestAUC = 0
  maxAUC = 0
  
  # initialize final feature list to all features in data
  finalFeatureList = colnames(traindata[, -ncol(traindata)])
  
  repeat
  { 	      	
    svmModel <- ksvm(class_label~., data = traindata, 
                     kernel = "rbfdot", type = "C-svc", C = 1, 
                     gamma = 1/ncol(traindata)-1, prob.model = TRUE, 
                     scaled = append(rep(TRUE, ncol(traindata)-1), FALSE))
    
    
    # predict probabilities on tuning set.
    p <- kernlab::predict(svmModel, tunedata, type = "probabilities")
    
    pred <- prediction(predictions = as.matrix(p[,2]),
                       labels = as.numeric(as.matrix(tunedata[,ncol(tunedata)])))
    
    # calculate AUC on tuning set.
    perf.auc <- performance(pred, measure = "auc")
    AUC <- unlist(perf.auc@y.values)
    cat("current auc: ", AUC, "\n")
    
    # check if AUC is greater than bestAUC
    if(AUC <= (0.95*maxAUC))
    { 		    	
      # stop timer
      runResults = proc.time() - runtime
      
      # return the model to use in calculating final AUC and return feature list
      model = list(AUC = bestAUC, Features = finalFeatureList, runResults = runResults);
      return(model) 
    }
    else
    {
      # record the current max AUC and the features that were used
      bestAUC = AUC
      finalFeatureList = colnames(traindata[, -ncol(traindata)])
      if(bestAUC >= maxAUC)
      {
        maxAUC = bestAUC
      }
    }
    
    # implement RFEST
    # create an array to store the list of differences in AUC for each feauture
    aucForEachFeature = c()
    
    # iterate through each feature
    # note: Do not go to the last feature (i.e. the class label)
    for(j in 1:(ncol(tunedata)-1))
    {
      # save copy of original feature vector
      original = tunedata[,j]
      
      # if continuous, permute. otherwise, flip
      if(length(unique(tunedata[,j])) > 2){
        tunedata[,j] = sample(tunedata[,j], nrow(tunedata), replace = FALSE, 
                              prob = rep(1/length(tunedata[,j]), nrow(tunedata)))
      } else{
        uniqueVals = unique(tunedata[,j])
        # if uniqueVals only has one value (due to near zero variance), replace with opposite number
        if(length(uniqueVals) == 1){
          tunedata[,j] = ifelse(tunedata[,j] == 0, 1, 0)
        } else{
          tunedata[,j] = ifelse(tunedata[,j] == uniqueVals[1], uniqueVals[2], uniqueVals[1])
        }      
      }
      
      # calculate new probabilities on testing set.
      p <- kernlab::predict(svmModel, tunedata, type = "probabilities")
      
      pred <- prediction(predictions = as.matrix(p[,2]),
                         labels = as.numeric(as.matrix(tunedata[,ncol(tunedata)])))
      perf.auc <- performance(pred, measure = "auc")
      AUCflipped <- unlist(perf.auc@y.values)
      
      # take difference between AUC_flipped and AUC_original.
      # note: the more negative, the more relevant
      diffAUC <- AUCflipped - AUC
      
      # Add diffAUC to aucForEachFeature
      aucForEachFeature = c(aucForEachFeature, diffAUC)
      
      # flip the value of the jth feature vector
      # in testdata back to its original value
      tunedata[,j] = original
      
      # repeat for the next feature.
    }
    
    # get class label vector for each dataset
    traindataClassLabel = traindata[,ncol(traindata)]
    tunedataClassLabel = tunedata[,ncol(tunedata)]
    
    # create temporary datasets for ordering by diffAUC
    # these datasets do not contain the class label
    traindataTemp = traindata[,-ncol(traindata)]
    tunedataTemp = tunedata[,-ncol(tunedata)]
    
    # combine temporary datasets with aucForEachFeature
    traindataTemp = rbind(traindataTemp, aucForEachFeature)  
    tunedataTemp = rbind(tunedataTemp, aucForEachFeature) 	
    
    # order tunedata and traindata in descending order.
    traindataTemp = traindataTemp[,order(-traindataTemp[nrow(traindataTemp),])]
    tunedataTemp = tunedataTemp[,order(-tunedataTemp[nrow(tunedataTemp),])]
    
    # remove given percent of features
    # record the number of items to delete
    numFeaturesToDelete = ceiling((percentRemoval/100)*(ncol(traindata)-1))
    
    # remove the least relevant feature(s) from datasets
    traindataTemp[1:numFeaturesToDelete] = list(NULL)
    tunedataTemp[1:numFeaturesToDelete] = list(NULL)
    
    # if there is 1 variable left, R will create a vector of the temporary data
    # prevent this from happening...
    if(length(colnames(traindataTemp)) == 1)
    {
      # get a copy of the variable name
      varName = colnames(traindataTemp)
      
      # remove the last row from datasets
      traindataTemp = traindataTemp[-nrow(traindataTemp),]
      tunedataTemp = tunedataTemp[-nrow(tunedataTemp),]
      
      # change back to dataframe and change the name of the column to varName
      traindataTemp = data.frame(traindataTemp)
      tunedataTemp = data.frame(tunedataTemp)
      
      colnames(traindataTemp) = varName
      colnames(tunedataTemp) = varName
    }
    else
    {
      # remove the last row from datasets
      traindataTemp = traindataTemp[-nrow(traindataTemp),]
      tunedataTemp = tunedataTemp[-nrow(tunedataTemp),] 
    }
    
    # make sure the class labels are factored
    traindata = cbind(traindataTemp, class_label = traindataClassLabel)
    tunedata = cbind(tunedataTemp, class_label = tunedataClassLabel)
  }
}


# fsAnalysis implementation is the function that you will run
# to find feature subset and run time
fsAnalysis <- function(dataSet, percentRemoval, algorithm, seed)
{
  # make class label values binary
  dataSet = binaryData(dataSet)
  
  # create 5 folds with createFolds()
  # note: 5 folds created for a 80/20 split (i.e. use fold 5 to tune)
  # for a 50/50 split, create 2 folds (then tune on fold 2)
  folds = createFolds(dataSet, 5, seed)
  
  # apply parallel processing for the number of outerfolds
  fsResults = RFEST_Features(dataSet = dataSet, percentRemoval = percentRemoval, folds = folds, foldNumber = 5)
  
  # extract results
  auc = fsResults$AUC
  featuresList = fsResults$Features
  runtime = fsResults$runResults[3]

  # create final results list for k fold cv
  result = list(AUC = auc, Features = featuresList, RunTime = runtime)
  
  return(result)
  
}










