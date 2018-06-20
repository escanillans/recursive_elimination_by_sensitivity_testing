# **************************************************************************
# Written By: Niko Escanilla
# Date: 05/16/18
# Subject: Main Program 
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
numFolds = 10
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

		# perform 10 fold cv
		models = Main(file, numFolds, percents[j], algorithm, seed)

		################# Compute Cross-Validated AUC ################# 

		# 1) Determine the average AUC and average AUC before breaking loop
		avgAUC = mean(models$AUCs)
    	avgAUC_before_break = mean(models$AUCs_before_break)
		cat("Final avg AUC: ", avgAUC, "\n")
		cat("Final highest avg AUC (before breaking) on tuning: ", avgAUC_before_break, "\n")

		stats = data.frame(avgAUC, mean(models$RunTimes))
		write.csv(stats, file = paste("results/", percents[j], "PercentRemoval/resultsOnData", 
			algorithm, "8020SplitSeed", seed, ".csv", sep = ""))

		# Save the list of AUCs
		write.csv(models$AUCs, file = paste("results/", percents[j], "PercentRemoval/aucForEachModel", 
			algorithm, "8020SplitSeed", seed, ".csv", sep = ""))

		# Save the list of run times
		write.csv(models$RunTimes, file = paste("results/", percents[j], "PercentRemoval/runTimeForEachModel", 
			algorithm, "8020SplitSeed", seed, ".csv", sep = ""))
		
		# Save list of AUCs (on tuning) before breaking loop
		write.csv(models$AUCs_before_break, file = paste("results/", percents[j], "PercentRemoval/highest_aucForEachModel", 
		                                   algorithm, "8020SplitSeed", seed, ".csv", sep = ""))
		
		################# Determine final feature set ################# 
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




