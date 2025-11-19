# MovieLens Capstone — Final Script
# -----------------------------------------------------------
# Outputs:
#  - Prints dev RMSEs for baseline/effects/MF/ensemble
#  - Prints final RMSE on final_holdout_test
#  - Writes results_movielens_final_predictions.csv

suppressPackageStartupMessages({
  if (!require(tidyverse)) install.packages("tidyverse", repos="http://cran.us.r-project.org")
  if (!require(caret))      install.packages("caret", repos="http://cran.us.r-project.org")
  if (!require(recosystem)) install.packages("recosystem", repos="http://cran.us.r-project.org")
  library(tidyverse); library(caret); library(recosystem); library(stringr)
})

seed_ml <- function(s) {
  if (getRversion() >= "3.6.0") set.seed(s, sample.kind = "Rounding") else set.seed(s)
}
DEV <- FALSE  # TRUE = fast draft runs; FALSE = full training for submission

RMSE <- function(y, yhat) sqrt(mean((y - yhat)^2))
clip_rating <- function(x) pmin(pmax(x, 0.5), 5)
extract_primary <- function(g) str_split_fixed(g, "\\|", 2)[,1]

# --- Data download & assembly (relative paths) ---
options(timeout = 600)
dir.create("data", showWarnings = FALSE)
zip_path <- file.path("data", "ml-10m.zip")
if (!file.exists(zip_path)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", destfile = zip_path, mode = "wb")
}
if (!dir.exists(file.path("data", "ml-10M100K"))) unzip(zip_path, exdir = "data")

ratings_file <- file.path("data", "ml-10M100K", "ratings.dat")
movies_file  <- file.path("data", "ml-10M100K", "movies.dat")

ratings <- readr::read_lines(ratings_file) %>%
  str_split_fixed("::", 4) %>% as.data.frame(stringsAsFactors = FALSE) %>%
  setNames(c("userId","movieId","rating","timestamp")) %>%
  mutate(across(c(userId, movieId, timestamp), as.integer), rating = as.numeric(rating))

movies <- readr::read_lines(movies_file) %>%
  str_split_fixed("::", 3) %>% as.data.frame(stringsAsFactors = FALSE) %>%
  setNames(c("movieId","title","genres")) %>% mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")
rm(ratings, movies)

# --- Construct edx and final_holdout_test ---
seed_ml(1)
idx <- createDataPartition(movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-idx, ]; temp <- movielens[idx, ]
final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")
removed <- anti_join(temp, final_holdout_test)
edx <- bind_rows(edx, removed)
rm(temp, removed, idx, movielens)

# --- Dev split (for model selection only) ---
seed_ml(42)
dev_idx   <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-dev_idx, ]
edx_test  <- edx[ dev_idx, ] %>%
  semi_join(edx_train, by = "movieId") %>% semi_join(edx_train, by = "userId")
rm(dev_idx)

# --- Baseline ---
mu_tr   <- mean(edx_train$rating)
rmse_mu <- RMSE(edx_test$rating, rep(mu_tr, nrow(edx_test)))
cat(sprintf("RMSE (Global mean): %.5f\n", rmse_mu))

# --- Regularized effects (movie, user, primary-genre) ---
set.seed(123)
edx_train <- edx_train %>% mutate(genre1 = extract_primary(genres))
edx_test  <- edx_test  %>% mutate(genre1 = extract_primary(genres))

lambda_grid <- c(1, 2, 3, 5, 7, 10, 15, 25)
score_lambda <- function(lambda){
  mu <- mean(edx_train$rating)
  b_i <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + lambda), .groups="drop")
  b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n() + lambda), .groups="drop")
  b_g <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>%
    group_by(genre1) %>% summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda), .groups="drop")
  pred <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_g, by="genre1") %>%
    transmute(pred = clip_rating(mu + coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0))) %>% pull(pred)
  RMSE(edx_test$rating, pred)
}
rmse_by_lambda <- sapply(lambda_grid, score_lambda)
lambda_best <- lambda_grid[which.min(rmse_by_lambda)]
mu <- mean(edx_train$rating)
b_i <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n() + lambda_best), .groups="drop")
b_u <- edx_train %>% left_join(b_i, by="movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n() + lambda_best), .groups="drop")
b_g <- edx_train %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% group_by(genre1) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n() + lambda_best), .groups="drop")

pred_eff_dev <- edx_test %>% left_join(b_i, by="movieId") %>% left_join(b_u, by="userId") %>% left_join(b_g, by="genre1") %>%
  transmute(pred = clip_rating(mu + coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0))) %>% pull(pred)
rmse_eff_dev <- RMSE(edx_test$rating, pred_eff_dev)
cat(sprintf("RMSE (Reg. effects, lambda=%g): %.5f\n", lambda_best, rmse_eff_dev))

# --- MF on dev ---
seed_ml(7)
train_file <- tempfile(); test_file <- tempfile()
edx_train %>% select(userId, movieId, rating) %>% write.table(train_file, sep=" ", row.names=FALSE, col.names=FALSE)
edx_test  %>% select(userId, movieId)        %>% write.table(test_file,  sep=" ", row.names=FALSE, col.names=FALSE)

r_dev <- Reco()
if (DEV) {
  tune_opts <- list(dim = c(20), lrate = c(0.05), costp_l2 = c(0.1), costq_l2 = c(0.1),
                    nthread = max(1, parallel::detectCores()-1), niter = 10)
  final_niter <- 20
} else {
  tune_opts <- list(dim = c(20, 40, 60), lrate = c(0.05, 0.1), costp_l2 = c(0.01, 0.1, 0.5),
                    costq_l2 = c(0.01, 0.1, 0.5), nthread = max(1, parallel::detectCores()-1), niter = 40)
  final_niter <- 80
}
tuned <- r_dev$tune(data_file(train_file), opts = tune_opts)
r_dev$train(data_file(train_file), opts = c(tuned$min, niter = final_niter, nthread = max(1, parallel::detectCores()-1)))

pred_file <- tempfile()
r_dev$predict(data_file(test_file), out_file(pred_file))
pred_mf_dev <- scan(pred_file); pred_mf_dev <- clip_rating(pred_mf_dev)
rmse_mf_dev <- RMSE(edx_test$rating, pred_mf_dev)
cat(sprintf("RMSE (MF tuned): %.5f\n", rmse_mf_dev))

# --- Ensemble on dev ---
fit <- lm(edx_test$rating ~ 0 + pred_eff_dev + pred_mf_dev)
w <- coef(fit)
w_eff <- unname(w["pred_eff_dev"]); w_mf <- unname(w["pred_mf_dev"])
w_eff <- ifelse(is.na(w_eff) || w_eff < 0, 0, w_eff)
w_mf  <- ifelse(is.na(w_mf)  || w_mf  < 0, 0, w_mf)
if (w_eff + w_mf == 0) { w_eff <- 0.5; w_mf <- 0.5 } else { s <- w_eff + w_mf; w_eff <- w_eff/s; w_mf <- w_mf/s }
cat(sprintf("Ensemble weights: w_eff=%.4f, w_mf=%.4f\n", w_eff, w_mf))

pred_ens_dev <- clip_rating(w_eff*pred_eff_dev + w_mf*pred_mf_dev)
rmse_ens_dev <- RMSE(edx_test$rating, pred_ens_dev)
cat(sprintf("RMSE (Ensemble effects+MF): %.5f\n", rmse_ens_dev))

# --- Final training on FULL edx and holdout evaluation ---
edx <- edx %>% mutate(genre1 = extract_primary(genres))
mu_full <- mean(edx$rating)

b_i_full <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_full)/(n() + lambda_best), .groups="drop")
b_u_full <- edx %>% left_join(b_i_full, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - mu_full - b_i)/(n() + lambda_best), .groups="drop")
b_g_full <- edx %>% left_join(b_i_full, by="movieId") %>% left_join(b_u_full, by="userId") %>% group_by(genre1) %>% summarize(b_g = sum(rating - mu_full - b_i - b_u)/(n() + lambda_best), .groups="drop")

predict_effects <- function(df){
  df %>% mutate(genre1 = extract_primary(genres)) %>%
    left_join(b_i_full, by="movieId") %>% left_join(b_u_full, by="userId") %>% left_join(b_g_full, by="genre1") %>%
    transmute(pred = clip_rating(mu_full + coalesce(b_i,0) + coalesce(b_u,0) + coalesce(b_g,0))) %>% pull(pred)
}

full_train_file <- tempfile(); holdout_uim_file <- tempfile(); final_pred_file <- tempfile()
edx %>% select(userId, movieId, rating) %>% write.table(full_train_file, sep=" ", row.names=FALSE, col.names=FALSE)
final_holdout_test %>% select(userId, movieId) %>% write.table(holdout_uim_file, sep=" ", row.names=FALSE, col.names=FALSE)

r_full <- Reco()
r_full$train(data_file(full_train_file), opts = c(tuned$min, niter = final_niter, nthread = max(1, parallel::detectCores()-1)))
r_full$predict(data_file(holdout_uim_file), out_file(final_pred_file))
final_pred_mf <- scan(final_pred_file); final_pred_mf <- clip_rating(final_pred_mf)

final_pred_eff <- predict_effects(final_holdout_test)
final_pred <- clip_rating(w_eff*final_pred_eff + w_mf*final_pred_mf)

# Safety checks
stopifnot(length(final_pred_mf) == nrow(final_holdout_test))
stopifnot(length(final_pred)    == nrow(final_holdout_test))

final_rmse <- RMSE(final_holdout_test$rating, final_pred)
cat(sprintf("\nFINAL RMSE on final_holdout_test: %.5f\n", final_rmse))

final_results <- final_holdout_test %>% select(userId, movieId, rating) %>% mutate(predicted = final_pred)
readr::write_csv(final_results, file.path("results_movielens_final_predictions.csv"))

# End of script
