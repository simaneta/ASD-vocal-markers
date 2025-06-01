partition_by_gender <- function(demo, features, hold_size = 0.3, gender_group = "Mixed",
                                male_test_ids = NULL,
                                female_test_ids = NULL,
                                male_train_ids = NULL,
                                female_train_ids = NULL) {
  set.seed(123) 
  
  demo <- demo %>% mutate(ID = as.character(ID))
  features <- features %>% mutate(ID = as.character(ID))
  
  # stratify based on male/female/mixed
  if (gender_group %in% c("Male", "Female")) {
    
    demo <- demo %>% filter(Gender == gender_group)
    
    # Compute n per class for hold-out
    n <- round((nrow(demo) * hold_size) / 2, 0)
    
    asd_sample <- demo %>%
      filter(Diagnosis == "ASD") %>%
      sample_n(n)
    
    td_sample <- demo %>%
      filter(Diagnosis == "TDC") %>%
      sample_n(n)
    
    hold_out <- bind_rows(asd_sample, td_sample)
    train <- demo %>% filter(!(ID %in% hold_out$ID))
    
  } else if (gender_group == "Mixed") {
    
    male_train_ids <- as.character(male_train_ids)
    female_train_ids <- as.character(female_train_ids)
    male_test_ids <- as.character(male_test_ids)
    female_test_ids <- as.character(female_test_ids)

    mixed_train_ids <- as.character(union(male_train_ids, female_train_ids))
    mixed_test_ids  <- as.character(union(male_test_ids, female_test_ids))
    
    train <- demo %>% filter(ID %in% mixed_train_ids)
    hold_out <- demo %>% filter(ID %in% mixed_test_ids)
    
  } else {
    stop("Invalid gender_group value.")
  }
  
  # Feature data subset
  train_features <- features %>% filter(ID %in% train$ID)
  hold_out_features <- features %>% filter(ID %in% hold_out$ID)
  
  return(list(hold_out_features, train_features))
}
