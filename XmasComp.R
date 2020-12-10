#########
library(tidymodels)      # for the recipes package, along with the rest of tidymodels
# Helper packages
library(nycflights13)    # for flight data
library(skimr)   
numcores <- parallel::detectCores()

set.seed(123)

xmas_data <- readr::read_csv("hackathon_training_data.csv")

xmas_data_quick <-
  xmas_data %>%
  select(-Response_binary, -Signature.14) %>%
  mutate_if(is.numeric, scale) %>%
  mutate_if(is.character, as.factor) %>%
  mutate(across(where(is.numeric), ~replace_na(.x, 0))) %>%
  mutate(across(where(is.logical), ~replace_na(.x, FALSE)))
  
skim(xmas_data_quick)

library(corrr)
# importnat
xmas_data_quick  %>% 
  mutate(response = as.numeric(response)) %>%
  dplyr::select_if(is.numeric) %>%
  correlate() %>%
  focus(response) %>%
  arrange(desc(abs(response)))

# unimportant
xmas_data_quick  %>% 
  mutate(response = as.numeric(response)) %>%
  dplyr::select_if(is.numeric) %>%
  correlate() %>%
  focus(response) %>%
  arrange(abs(response))
  


# training test split
# NB strata = TUMOURTYPE for later
data_split <- initial_split(xmas_data_quick, strata = histology, prop = 3/4)
train_data <- training(data_split)
test_data  <- testing(data_split)

# NB check that the test data doesn't lose a whole tumour type or grade or something
train_data %>% 
  distinct(histology) %>% 
  anti_join(train_data)

# 5 fold CV splits Data is too bif of r10 fols with tumour strata
cv_splits <- vfold_cv(train_data, v = 5)

# recipe for data preprocessing
preprocess <- 
  recipe(response ~ ., data = train_data) %>% 
  update_role(case, new_role = "ID") %>% 
  step_zv(all_predictors())

# Logistic regression elastic net i.e BOTH penalty L2 and mixture L1
logit_tune <- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")
# Hyperparameter grid
logit_grid <- logit_tune %>%
  parameters() %>%
  grid_max_entropy(size = 6)
# Workflow bundling every step 
logit_wflow <- workflow() %>%
  add_recipe(preprocess) %>%
  add_model(logit_tune)
logit_res <- 
   logit_wflow %>% 
   tune_grid(resamples = cv_splits, grid = logit_grid)


# random forest
# rf_tune <- rand_forest(mtry = tune(), trees = tune()) %>%
#   set_engine("ranger") %>%
#   set_mode("classification")
# rf_grid <- rf_tune %>%
#   parameters() %>%
#   finalize(select(train_data, -arr_delay)) %>%
#   grid_max_entropy(size = 6)
# rf_wflow <- workflow() %>%
#   add_recipe(preprocess) %>%
#   add_model(rf_tune)
# rf_res <-
#    rf_wflow %>%
#    tune_grid(resamples = cv_splits, grid = rf_grid)



#boosted trees
# boost_tune <- boost_tree(mtry = tune(), tree = tune(),
#                   learn_rate = tune(), tree_depth = tune()) %>%
#   set_engine("xgboost") %>%
#   set_mode("classification")
# boost_grid <- boost_tune %>%
#   parameters() %>%
#   finalize(select(train_data, -arr_delay)) %>%
#   grid_max_entropy(size = 6)
# boost_wflow <- workflow() %>%
#   add_recipe(preprocess) %>%
#   add_model(boost_tune)
# boost_res <-
#   boost_wflow %>%
#   tune_grid(resamples = cv_splits, grid = boost_grid)


#neural nets
# keras_tune <- mlp(hidden_units = tune(), penalty = tune(), activation = "relu") %>%
#   set_engine("keras") %>%
#   set_mode("classification")
# keras_grid <- keras_tune %>%
#   parameters() %>%
#   grid_max_entropy(size = 6)
# keras_wflow <- workflow() %>%
#   add_recipe(preprocess) %>%
#   add_model(keras_tune)

# keras_res <- 
#    keras_wflow %>% 
#    tune_grid(resamples = cv_splits, grid = keras_grid)


logit_best <- 
  logit_res %>% 
  select_best(metric = "roc_auc")

final_logit_wflow <- 
  boost_wflow %>% 
  finalize_workflow(boost_best)

final_logit <- 
  final_logit_wflow %>%
  fit(data = train_data) 

logit_best <- 
  logit_res %>% 
  select_best(metric = "roc_auc")

final_logit_wflow <- 
  logit_wflow %>% 
  finalize_workflow(logit_best)

# best boost fit on all the train data (no folding)
final_logit <- 
  final_boost_wflow %>%
  fit(data = train_data) 

library(vip)
final_logit %>% 
  pull_workflow_fit() %>% 
  vip()

# Now fit the final model on all test + train data
final_logit_fit <- 
  final_logit %>%
  last_fit(data_split) 

# FINALLY! How did we do?
final_logit_fit %>%
  collect_metrics()


# final_boost_wflow <- 
#   boost_wflow %>% 
#   finalize_workflow(boost_best)
# 
# final_boost <- 
#   final_boost_wflow %>%
#   fit(data = train_data) 
# 
# boost_best <- 
#   boost_res %>% 
#   select_best(metric = "roc_auc")
# 
# final_boost_wflow <- 
#   boost_wflow %>% 
#   finalize_workflow(boost_best)
# 
# # best boost fit on all the train data (no folding)
# final_boost <- 
#   final_boost_wflow %>%
#   fit(data = train_data) 

# which variables are important? 
# library(vip)
# final_boost %>% 
#   pull_workflow_fit() %>% 
#   vip()

# # Now fit the final model on all test + train data
# final_boost_fit <- 
#   final_boost %>%
#   last_fit(data_split) 
# 
# # FINALLY! How did we do?
# final_boost_fit %>%
#   collect_metrics()


 
final_boost_fit %>%
  collect_predictions() %>% 
  roc_curve(.pred_class, arr_delay) %>% 
  autoplot()

# # SPLIT the running across 4 jobs fo r4 cores more if I can get a better computer
# wflow_list <- list(logit_wflow, rf_wflow, boost_wflow)
# grid_list <- list(logit_grid, rf_grid, boost_grid)
# #ctrl_grid = stacks::control_stack_grid()
# future::plan("multicore", workers = numcores)
# Sys.time()
# 
# trained_models_list <- furrr::future_map2(.x = wflow_list, .y = grid_list, ~tune_grid(.x , resamples = cv_splits, grid = .y))
# 
# Sys.time()
# 
# 
# # select not the best model but nearly the best (with 1SE) with less complexity (penalty in this example)
# logit_res %>% select_by_one_std_err(metric = "roc_auc", desc(penalty))
# 
# data_stack <- stacks::stacks() %>%
#   stacks::add_candidates(logit_res) %>%
#   stacks::add_candidates(boost_res)
