# **************************************************************************
# Written By: Niko Escanilla
# Subject: Main Program for RFEST
#***************************************************************************
rm(list = ls())

# Perform RFEST Analysis
# Run script "rfestPackages.R" to install necessary packages
source("rfestPackages.R")

# Load in data and set parameters for RFEST
# 10 fold cv, 10 percent removal, seed 1
file = read.csv("data/cleanData/final_data.csv")

# set parameters
algorithm = "RFEST"
percents = c(10)
seeds = c(1)

count = 0
for(seed in seeds)
{
	for(j in 1:length(percents))
	{
		count = count + 1
		cat("Currently working on file: ", count, "/", length(seeds)*length(percents), "\n")	
		cat("Seed: ", seed, ". Percent removal: ", percents[j], "\n")

		source("functions.R")

		################# Find subset of relevant features to learning task ################# 
		# Run RFEST on train and tune...final features are whatever is at the end when AUC drops
		cat("Determining best set of features to retain...", "\n")
		fsResults = fsAnalysis(file, percents[j], algorithm, seed)

		# 1) Determine the AUC achieved before breaking loop and runtime
		AUC_before_break = fsResults$AUC
		runtime = fsResults$RunTime

		# Save above as a dataset
		stat = data.frame(AUC_before_break, runtime)
		write.csv(stat, file = paste("results/", percents[j], "PercentRemoval/fsResultsAucAndRuntime", 
			algorithm, "8020SplitSeed", seed, ".csv", sep = ""))

		# Save list of features
		fsDataFrame = data.frame(Features = rev(fsResults$Features))
		write.csv(fsDataFrame, file = paste("results/", percents[j], "PercentRemoval/fsListOfFeatures", 
			algorithm, "8020SplitSeed", seed, ".csv",sep = ""))
	}
}




