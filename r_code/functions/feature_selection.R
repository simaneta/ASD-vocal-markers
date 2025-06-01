elastic <- function(train_data, 
                    folds, 
                    id_col, 
                    featureset = featureset, 
                    gender_group,
                    hold_set,
                    demo_set) {
  
  library(glue) 
  
  # Partition using groupdata2, keeping IDs in same fold
  fold_train <- train_data %>% 
    groupdata2::fold(k = folds, id_col = id_col)  # adds `.folds` column
  
  feature_list <- NULL 
  
  for (i in unique(fold_train$.folds)) {
    message(glue("Now looping through fold {i}"))
    
    lasso_train <- fold_train %>% filter(.folds != i)
    lasso_train <- lasso_train %>% dplyr::select(-any_of(id_col))
    
    # Prepare training matrix
    y <- lasso_train$Diagnosis
    
    cols_to_drop <- c("Diagnosis", ".folds", "Gender", "feature_set", "ID", "country")
    lasso_data <- lasso_train %>% dplyr::select(-any_of(cols_to_drop))  # EXCLUDE Diagnosis
    
    x <- model.matrix(~ ., data = lasso_data)
    
    message(glue("Fold {i}: Diagnosis levels = {paste(levels(y), collapse = ', ')}"))
    
    
    # Lasso feature selection
    set.seed(123) 
    cv_lasso <- cv.glmnet(x,
                          y,
                          alpha = 0.5, 
                          standardize = FALSE,
                          family = "binomial",
                          type.measure = "auc")
    
    
    # extract non-zero features at lambda.1se
    lasso_coef <- tidy(cv_lasso$glmnet.fit) %>%
      filter(lambda == cv_lasso$lambda.1se,
             term != "(Intercept)") %>%
      mutate(abs = abs(estimate)) %>%
      filter(abs > 0)
    
    # fallback to lambda.min + top 30 features
    if (nrow(lasso_coef) == 0) {
      lasso_coef <- tidy(cv_lasso$glmnet.fit) %>%
        filter(lambda == cv_lasso$lambda.min,
               term != "(Intercept)") %>%
        mutate(abs = abs(estimate)) %>%
        arrange(desc(abs)) %>%
        slice(1:30)
    }
    
    selected <- lasso_coef %>%
      filter(!str_detect(term, ".folds")) %>%
      dplyr::select(term) %>%
      rename(features = term) %>%
      mutate(fold = as.character(i))
    
    print(glue("Fold {i}: non-zero features = {nrow(lasso_coef)}"))
    
    feature_list <- bind_rows(feature_list, selected)
  }
  
  # only keep features present in 2+ folds
  stable_features <- feature_list %>%
    group_by(features) %>%
    filter(n() >= ceiling(folds / 2)) %>% 
    ungroup() %>%
    distinct(features) %>%
    mutate(fold = "stable")
  
  print(glue("Fold {i}: non-zero stable features = {nrow(stable_features)}"))
  
  # save feature sets
  write.csv(fold_train, glue("../data/speech_data/data_{featureset}_{tolower(gender_group)}.csv"), row.names = FALSE)
  write.csv(feature_list, glue("../data/feature_lists/features_{featureset}_{tolower(gender_group)}.csv"), row.names = FALSE)
  write.csv(stable_features, glue("../data/feature_lists/features_{featureset}_{tolower(gender_group)}_STABLE.csv"), row.names = FALSE)
}
