# ***************************************************************************
# Written By: Niko Escanilla
# Date: 05/19/17
# Subject: Run linear SVM on final set of features 
#***************************************************************************
rm(list = ls())

# Run script "rfestPackages.R" to install necessary packages
source("rfestPackages.R")

# Load "functions.R" that contains stratified cv
set.seed(1)
source("functions.R")

# Load the set of features retained from RFEST 10% removal seed 1
listOfFeatures = read.csv("results/10PercentRemoval/fsListOfFeaturesRFEST8020SplitSeed1.csv")

# Remove first column (i.e. feature number)
listOfFeatures[,1] = NULL

# Load the complete dataset
data = read.csv("data/cleanData/final_data.csv")

# Get list of features retained
finalListOfFeatures = c()
for(i in 1:nrow(listOfFeatures))
{
  finalListOfFeatures = c(finalListOfFeatures, as.character(listOfFeatures[i,]))
}

# Create dataset that contains only features retained, interaction terms, and class labels
finalDataSet = data[,finalListOfFeatures]

# Create all interaction terms for listOfFeatures
for(i in 1:(nrow(listOfFeatures)-1))
{
  # Get the ith feature name
  ithFeatureName = as.character(listOfFeatures[i,])
  
  for(j in (i+1):nrow(listOfFeatures))
  {
    # Get the jth 
    jthFeatureName = as.character(listOfFeatures[j,])
    
    # Calculate interaction term
    interactionTerm = data[,ithFeatureName]*data[,jthFeatureName]
    
    # Add interaction term to finalDataSet
    finalDataSet = cbind(finalDataSet, interactionTerm)
    names(finalDataSet)[ncol(finalDataSet)] = as.character(paste(ithFeatureName,":",jthFeatureName,sep = ""))
  }
}

# Add class label to dataset
finalDataSet = cbind(finalDataSet, data$class_label)
names(finalDataSet)[ncol(finalDataSet)] = "class_label"

# Create model on training set. 
cat("building svm model", "\n")
finalModel <- svm(class_label~., data=finalDataSet, type="C-classification", 
                  kernel="linear", cost=1, scale=append(rep(TRUE, ncol(finalDataSet)-1), FALSE))

interaction.weights <- t(finalModel$coefs) %*% finalModel$SV
interaction.weights <- t(interaction.weights)
dfInteractionWeights = data.frame(Features = row.names(interaction.weights), Weights = interaction.weights[,1])
row.names(dfInteractionWeights)  = NULL

# remove X. from interaction pairs
dfInteractionWeights$Features = gsub("X.", "", dfInteractionWeights$Features) 

# save csv
write.csv(dfInteractionWeights,file="results/10PercentRemoval/interaction.weights.8020Split.csv")

# Order dfInteractionWeights by absolute value
dfInteractionWeightsAbsVal = data.frame(Features = row.names(interaction.weights), WeightsAbsVal = abs(dfInteractionWeights$Weights))
dfInteractionWeightsAbsVal = dfInteractionWeightsAbsVal[order(-dfInteractionWeightsAbsVal$WeightsAbsVal),]
dfInteractionWeightsAbsVal$Features = gsub("X.", "", dfInteractionWeightsAbsVal$Features)
write.csv(dfInteractionWeightsAbsVal, file = "results/10PercentRemoval/interaction.weights.abs.valinteraction.weights.8020Split.csv")
cat("finished!", "\n")











