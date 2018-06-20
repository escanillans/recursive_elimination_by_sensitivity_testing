# ***************************************************************************
# Written By: Niko Escanilla
# Date: 11/05/16
# Subject: Install packages for RFEST algorithm
#***************************************************************************

# Download package dplyr for selecting subsets of data
if (!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
}


# Download package ksvm for ksvm function 
if (!require(kernlab)) {
  install.packages("kernlab")
  library(kernlab)
}

# Download package caret for tuning the svm
if (!require(caret)) {
  install.packages("caret")
  library(caret)
}


if(!require(e1071)){
	install.packages("e1071")
	library(e1071)
}

# Download package plyr for rbind.fill function
if (!require(plyr)) {
  install.packages("plyr")
  library(plyr)
}

# Download package ROCR for ROC and AUC performance measurements.
if (!require(ROCR)){
  install.packages("ROCR")
  library(ROCR)
}

# Download package parallel to run algorithm in parallel
if (!require(parallel)){
  install.packages("parallel")
  library(parallel)
}

if (!require(readxl)){
  install.packages("readxl")
  library(readxl)
}

# Download permute package for shuffling data that is not binary
#if(!require(permute)){
#  install.packages("permute")
#  library(permute)
#}

# Download mice for multiple imputation
#if (!require(mice)) {
#  install.packages("mice")
#  library(mice)
#}

