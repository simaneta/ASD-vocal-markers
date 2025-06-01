# Load libraries
library(tidyverse)
library(readr)
library(glmnet)
library(data.table)
library(broom)
library(forcats)
library(e1071)
library(cvms)
library(groupdata2)
library(dplyr)

# Load helper functions as needed
source("functions/new_partitioning.R")
source("functions/Normalize_function.R")
source("functions/feature_selection.R")
source("functions/combine_dfs.R")

grand_function_new <- function(features,
                                  demo,
                                  gender_group,
                                  featureset,
                                  n_lasso_folds = 3, 
                                  male_test_ids = NULL,
                                  female_test_ids = NULL,
                                  male_train_ids = NULL,
                                  female_train_ids = NULL) {
  
  set.seed(123) 
  
  # Filter by gender
  demo_filtered <- demo %>%
    filter(ID %in% features$ID) %>%
    distinct(ID, .keep_all = TRUE)
  
  # Combine features with demo data
  data_combined <- combined_data(data = features, demo = demo_filtered)
  
  # Partition: split into train/test by participant ID
  male_train_ids <- as.character(male_train_ids)
  female_train_ids <- as.character(female_train_ids)
  male_test_ids <- as.character(male_test_ids)
  female_test_ids <- as.character(female_test_ids)
  
  demo_filtered$ID <- as.character(demo_filtered$ID)
  data_combined$ID <- as.character(data_combined$ID)
  
  all_ids_used <- c(male_train_ids, male_test_ids, female_train_ids, female_test_ids)

  partitions <- partition_by_gender(demo = demo_filtered,
                                    features = data_combined,
                                    hold_size = 0.3,
                                    gender_group = gender_group,
                                    male_test_ids = male_test_ids,
                                    female_test_ids = female_test_ids,
                                    male_train_ids = male_train_ids,
                                    female_train_ids = female_train_ids)
  
  hold_out <- partitions[[1]] 
  train <- partitions[[2]] 
  
  train$ID <- as.factor(train$ID)
  hold_out$ID <- as.factor(hold_out$ID)
  
  ## Normalize (scale): fit scaler on train, apply to test
  train_scaled <- as.data.frame(scale_function(train, datatype = "train"))
  hold_out_scaled <- as.data.frame(scale_function(train, hold_out, datatype = "test"))
  hold_out_scaled$ID <- hold_out$ID
  
  ### Save datasets
  write.csv(train_scaled,
            paste0("../data/speech_data/", featureset, "_", tolower(gender_group), "_train.csv"),
            row.names = FALSE)
  
  write.csv(hold_out_scaled,
            paste0("../data/speech_data/", featureset, "_", tolower(gender_group), "_test.csv"),
            row.names = FALSE)
  
  ### Elastic net feature selection 
  train_scaled$ID <- as.factor(train_scaled$ID)
  train_scaled$Diagnosis <- as.factor(train_scaled$Diagnosis)
  
  elastic(train_data = train_scaled,
          folds = n_lasso_folds,
          id_col = "ID",
          featureset = featureset,
          gender_group = gender_group,
          hold_set = hold_out_scaled,
          demo_set = demo_filtered)
}
