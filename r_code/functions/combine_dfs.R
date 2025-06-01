combined_data <- function(data, demo) {
  # Only keep necessary columns from demo
  demo_clean <- demo %>% 
    select(ID, Diagnosis, Gender)
  
  # Ensure data doesn't have Diagnosis or Gender already
  data <- data %>% select(-any_of(c("Diagnosis", "Gender")))
  
  data <- data %>%
    left_join(demo_clean, by = "ID") %>%
    mutate(
      Diagnosis = as.factor(Diagnosis),
      Gender = as.factor(Gender)
    )
  
  return(data)
}
