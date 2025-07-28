# Lech Madeyski v.2024R1
here::i_am("R/LA2SDP.R")

gc(reset = TRUE)

# At the start of the R script, MadeyskiStradowski.R, we run set.seed(123) and
# use the R package renv to manage package versions.
set.seed(123)

# Readers may find our lock file in the reproduction package provided in the
# Supplementary Material. Reproducibility is often problematic when
# parallelization of computations is used, and we heavily used parallelization
# to speedup computations. To overcome this problem and support reprodicibility,
# we employed the future package that ensures that all workers will receive the
# same pseudorandom number generator streams, independent of the number of
# workers, see https://www.jottr.org/2020/09/22/push-for-statistical-sound-rng/

# We will use renv to manage packages in a project-specific way.
# To use renv to install packages, you will first need to install the renv package.
packages <- c("remotes", "renv")
renv::install(setdiff(packages, rownames(installed.packages()))) # install.packages(setdiff(packages, rownames(installed.packages())))


# We need mlr3 version 0.21.1 which has fixed error in the mlr3 package.
# renv::install("mlr3@0.21.1")
# Define the package name and desired version
package_name <- "mlr3"
desired_version <- "1.0.1"
# Check if the package is installed
if (requireNamespace(package_name, quietly = TRUE)) {
  # Get the currently installed version
  installed_version <- packageVersion(package_name)

  # Check if the installed version matches the desired version
  if (installed_version == desired_version) {
    cat("Package", package_name, "version", desired_version, "is already installed.\n")
  } else {
    cat("Package", package_name, "version", desired_version, "is not installed.\n")
    cat("Installed version:", unlist(installed_version), "\n")
    cat("Installing", package_name, "version", desired_version, "using renv...\n")

    # Install the desired version using renv
    renv::install(paste0(package_name, "@", desired_version))
  }
} else {
  cat("Package", package_name, "is not installed.\n")
  cat("Installing", package_name, "version", desired_version, "using renv...\n")

  # Install the desired version using renv
  renv::install(paste0(package_name, "@", desired_version))
}

# We’ll install the required packages from the usual CRAN repository:
requiredPackages <- c(
  "aorsf",
  # "apcluster",
  # "catboost",
  "checkmate", # provides assert functions, e.g., assertNumeric
  "crop",
  "DALEX",
  "DALEXtra",
  # "data.table",
  # "dictionar6",
  # "DiceKriging",
  "digest",
  "dplyr", # could be loaded via tidyverse
  # "earth",
  "forstringr",
  # "FSelectorRcpp", #needed to select features on a basis of information_gain
  "future",
  # "ggplot2", #loaded via tidyverse
  "here",
  "kernlab",
  "kknn",
  "lightgbm",
  # imports "Matrix" package
  "lgr",
  "lubridate",
  # "MASS",
  "Matrix",
  # "mlr3",
  "mlr3benchmark",
  # #"mlr3extralearners", #mlr3extralearners lives on GitHub and will not be on CRAN.
  "mlr3filters",
  "mlr3fselect",
  "mlr3hyperband",
  "mlr3learners",
  # #"mlr3mbo",
  "mlr3pipelines",
  # #will be installed w/o error: package 'Biobase' is not available
  # "mlr3proba", #renv::install("mlr3proba", repos = "https://mlr-org.r-universe.dev")
  # "mlr3temporal",
  "mlr3tuning",
  "mlr3tuningspaces",
  # "mlr3viz",
  # "mlr3verse",
  # "ooplah",
  "paradox",
  "parallelly",
  # "PMCMRplus",
  # "praznik",
  # "precrec",
  "progressr",
  "purrr",
  # "quanteda",
  # "stopwords",
  # "randomForestSRC",
  "ranger",
  # "rcompanion", #stat tests
  # "readr", #loaded via tidyverse
  "readxl",
  "remotes",
  # "reshape2",
  # "RWeka",
  # "R6",
  # "rgenoud",
  # "scriptName",
  "skimr",
  "text2vec",
  "tictoc",
  "tidyr",
  # "tidyverse",
  # "usethis",
  "writexl"
)
# install.packages(requiredPackages, repos = "https://cloud.r-project.org/", dependencies = TRUE)
# install.packages(setdiff(requiredPackages, rownames(installed.packages())),
renv::install(setdiff(requiredPackages, rownames(installed.packages())),
  repos = "https://cloud.r-project.org/",
  dependencies = TRUE
)
lapply(requiredPackages, require, character.only = TRUE)

# Take a snapshot to save the environment status:
# renv::snapshot()




# We may crawl our R files and look for package names, returning a list, via renv::dependencies("PATH")
# We may also use renv::restore() to restore the project to a previous state, or renv::status() to check the status of the project.
# We may also use renv::install() to install the packages listed in the lockfile, or renv::update() to update the packages listed in the lockfile.


# Install packages not available from CRAN
# latest GitHub release
# remotes::install_github("mlr-org/mlr3extralearners@*release", dependencies=TRUE)
# remotes::install_github("mlr-org/mlr3extralearners", dependencies=TRUE)
library(mlr3extralearners)

# install.packages("mlr3proba", repos = "https://mlr-org.r-universe.dev")
library(mlr3proba)

# Take a snapshot to save the environment status:
# renv::snapshot()


if (!require("mlr3extralearners")) {
  renv::install("mlr-org/mlr3extralearners@*release")
}

# Install some development packages from GitHub
# catboost installation on Mac, Linux etc. is OS specific
if (!require("catboost")) {
  switch(Sys.info()[["sysname"]],
    Windows = {
      print(
        "This final version of script was througly testes on Mac and earlier on Linux. It should also work fine on Windows if you install 'catboost'. You may use https://github.com/catboost/catboost/releases/download/v1.2.5/catboost-1.2.5.exe"
      )
    },
    # use https://github.com/catboost/catboost/releases/download/v1.2.2/catboost-1.2.2.exe
    Linux = {
      remotes::install_url(
        url = "https://github.com/catboost/catboost/releases/download/v1.2.5/catboost-R-Linux-1.2.5.tgz",
        INSTALL_opts = c(
          "--no-multiarch",
          "--no-test-load",
          "--no-staged-install"
        )
      )
    },
    Darwin = {
      remotes::install_url(
        url = "https://github.com/catboost/catboost/releases/download/v1.2.5/catboost-R-Darwin-1.2.5.tgz",
        INSTALL_opts = c(
          "--no-multiarch",
          "--no-test-load",
          "--no-staged-install"
        )
      )
    }
  )
  library(catboost)
}

# Take a snapshot to save the environment status:
# renv::snapshot()

# Collect Information About the Current R Session (R packages and their versions)
sessionInfo()

# Parallel Backend
# workers_to_use = 12 # M3 Max; it has 12 high-performance CPU cores, four efficiency cores, a 40-core GPU
workers_to_use <- 4 # M2 Pro; it has 4 high-performance CPU cores, four efficiency cores, a 19-core GPU
if (future::supportsMulticore()) {
  message("[multicore mode] cores: ", workers_to_use)
  # future::plan("multicore", workers = workers_to_use)
  message("but for the safety reason we stick to [multisession mode].")
  future::plan("multisession", workers = workers_to_use)
} else {
  message("[multisession mode] cores:", workers_to_use)
  future::plan("multisession", workers = workers_to_use)
}

progressr::handlers(global = TRUE)
progressr::handlers("progress")

tic()


# --------------------------------------- #
# Variables to control the script's flow:
# set TRUE or FALSE to decompose date into year, quarter, month, week, mday, wday
DATE_DECOMPOSED <- TRUE # FALSE
# set TRUE or FALSE to use persisted results or not
USE_PERSISTED_RESULTS <- FALSE
# --------------------------------------- #

# Data input
# Set the working directory to the location of the data files
# setwd("/Users/l...")

read_excel_dataset <- function(dataset_name) {
  dataset_filename <- paste0(dataset_name, ".xlsx")
  lgr$info(paste0("Read ", dataset_filename))
  dataset <- readxl::read_excel(dataset_filename)
}


preprocess_dataset <- function(dataset, drop_na = FALSE) {
  dataset <- as.data.frame(unclass(dataset), stringsAsFactors = TRUE)

  # https://learnetutorials.com/r-programming/date-time-handling
  # https://www.neonscience.org/resources/learning-hub/tutorials/dc-convert-date-time-posix-r

  dataset$EXECUTION.DATE <- as.POSIXct(dataset$EXECUTION.DATE)
  if (DATE_DECOMPOSED == TRUE) {
    # year, quarter, month are excluded as they lack predictive power in a "time-based data set split", which is the appropriate approach.
    dataset$week <- as.numeric(lubridate::week(dataset$EXECUTION.DATE))
    dataset$wday <- lubridate::wday(dataset$EXECUTION.DATE) # , label = TRUE)
  }
  # dataset$EXECUTION.DATE <- NULL
  dataset$EXECUTION.DATE <- as.numeric(dataset$EXECUTION.DATE)
  dataset <- as.data.frame(unclass(dataset), stringsAsFactors = TRUE)
  str(dataset$month)

  # colnames(dataset)
  dataset <- dplyr::select(
    dataset,
    -c(
      FAULT.REPORT.ID,
      FAULT.CREATION.DATE,
      DOMAIN,
      PROJECT,
      FAULT.REPORT.NB,
      AUTOMATION.LEVEL,
      DETAILED.AUTOMATION.LEVEL
    )
  )
  ### EDITS ----------------------

  # COL                       EXAMPLE
  #  ----------------------------------------------------------
  #' data.frame':   82032 obs. of  20 variables:
  # AUTOMATION.LEVEL.FINAL        : Factor w/ 2 levels "AUTOMATED","MANUAL": 1 1 1 1 1 1 1 1 1 1 ...
  # PROGRAM.PHASE                 : Factor w/ 61 levels "0.0PD","0.0PD internal",..: 1 1 36 36 36 1 1 1 1 1 ...
  # RELEASE                       : Factor w/ 10 levels "5G00","5G16",..: 1 1 9 9 9 1 8 1 1 1 ...
  # TEST.AUTOMATION.LEVEL         : Factor w/ 2 levels "AUTOMATED","MANUAL": 2 2 1 1 1 2 1 1 1 2 ...
  # TEST.ENTITY                   : Factor w/ 3 levels "CIT","CRT","Manual": 2 2 NA NA NA 2 NA NA NA 2 ...
  # TEST.OBJECT                   : Factor w/ 3 levels "Benchmark","New Feature",..: 3 3 2 2 2 3 3 3 3 3 ...
  # ORGANIZATION                  : Factor w/ 127 levels "4G_ASL2_WRO - SG3",..: 18 72 123 123 123 82 20 17 17 82 ...
  # week                          : num  1 1 1 1 1 1 1 1 1 1 ...
  # wday                          : num  6 6 6 6 6 1 1 2 2 2 ...
  # TEST.AUTOMATION.LEVELAUTOMATED: num  0 0 1 1 1 0 1 1 1 0 ...
  # TEST.AUTOMATION.LEVELMANUAL   : num  1 1 0 0 0 1 0 0 0 1 ...

  # 1. Combine selected text features
  dataset$text_combined <- paste(
    dataset$PROGRAM.PHASE,
    dataset$RELEASE,
    dataset$TEST.OBJECT,
    dataset$ORGANISATION,
    sep = "_"
  )

  # Use a hash function to create a unique numeric value
  # Use crc32 hash if you want integer, or md5 then convert
  dataset$text_hash <- sapply(dataset$text_combined, function(txt) {
    as.integer(substr(digest(txt, algo = "crc32", serialize = FALSE), 1, 8), 16L)
  })

  dataset$text_combined <- NULL

  # ✅ Optional sanity check
  cat("Final matrix dimensions:", paste(dim(dataset), collapse = " x "), "\n")
  print("Columns:")
  print(colnames(dataset))
  # print("Dataset:")
  # str(dataset)

  ### END OF ---------------------

  # Rev. 1 rightly suggested to remove TEST.RUN.ID and EXECUTION.DATE
  dataset <- dplyr::select(dataset, -c(TEST.RUN.ID, EXECUTION.DATE))

  if (drop_na == TRUE) {
    dataset <- dataset %>% tidyr::drop_na() # drop rows with NA values in any column
  }

  dataset <- dplyr::mutate(dataset, across(where(is.integer), as.numeric)) # needed for catboost

  dataset
}


QCdata_1 <- read_excel_dataset("QCdata_1")
QCdata_1 <- preprocess_dataset(QCdata_1)
# str(QCdata_1)
task_1 <- TaskClassif$new("QCdata_1", QCdata_1, "TEST.STATUS")

QCdata_2 <- read_excel_dataset("QCdata_2")
QCdata_2 <- preprocess_dataset(QCdata_2)
# str(QCdata_2)
task_2 <- TaskClassif$new("QCdata_2", QCdata_2, "TEST.STATUS")

QCdata_3 <- read_excel_dataset("QCdata_3")
QCdata_3 <- preprocess_dataset(QCdata_3)
# str(QCdata_3)
task_3 <- TaskClassif$new("QCdata_3", QCdata_3, "TEST.STATUS")

QCdata_4 <- read_excel_dataset("QCdata_4")
QCdata_4 <- preprocess_dataset(QCdata_4)
# str(QCdata_4)
task_4 <- TaskClassif$new("QCdata_4", QCdata_4, "TEST.STATUS")

QCdata_5 <- read_excel_dataset("QCdata_5")
QCdata_5 <- preprocess_dataset(QCdata_5)
# str(QCdata_5)
task_5 <- TaskClassif$new("QCdata_5", QCdata_5, "TEST.STATUS")

QCdata_6 <- read_excel_dataset("QCdata_6")
QCdata_6 <- preprocess_dataset(QCdata_6)
# str(QCdata_6)
task_6 <- TaskClassif$new("QCdata_6", QCdata_6, "TEST.STATUS")

QCdata_All <- rbind(QCdata_1, QCdata_2, QCdata_3, QCdata_4, QCdata_5, QCdata_6)
# str(QCdata_All)

task_all <- TaskClassif$new("QCdata_All", QCdata_All, "TEST.STATUS")

QCdata_1_2 <- rbind(QCdata_1, QCdata_2)
QCdata_1_3 <- rbind(QCdata_1, QCdata_2, QCdata_3)
QCdata_1_4 <- rbind(QCdata_1, QCdata_2, QCdata_3, QCdata_4)
QCdata_1_5 <- rbind(QCdata_1, QCdata_2, QCdata_3, QCdata_4, QCdata_5)
QCdata_1_6 <- rbind(QCdata_1, QCdata_2, QCdata_3, QCdata_4, QCdata_5, QCdata_6)

QCdata_2_3 <- rbind(QCdata_2, QCdata_3)
QCdata_2_4 <- rbind(QCdata_2, QCdata_3, QCdata_4)
QCdata_2_5 <- rbind(QCdata_2, QCdata_3, QCdata_4, QCdata_5)
QCdata_2_6 <- rbind(QCdata_2, QCdata_3, QCdata_4, QCdata_5, QCdata_6)

QCdata_3_4 <- rbind(QCdata_3, QCdata_4)
QCdata_3_5 <- rbind(QCdata_3, QCdata_4, QCdata_5)
QCdata_3_6 <- rbind(QCdata_3, QCdata_4, QCdata_5, QCdata_6)

QCdata_4_5 <- rbind(QCdata_4, QCdata_5)
QCdata_4_6 <- rbind(QCdata_4, QCdata_5, QCdata_6)

QCdata_5_6 <- rbind(QCdata_5, QCdata_6)


rsmp_repeated_cv_f5_r2 <- rsmp("repeated_cv", folds = 5, repeats = 2)
rsmp_cv10 <- rsmp("cv", folds = 10)
rsmp_cv3 <- rsmp("cv", folds = 3)
rsmp_holdout75 <- rsmp("holdout", ratio = 0.75)
msr_mcc <- msr("classif.mcc")
measures <- msrs(c("classif.mcc", "classif.acc", "classif.recall", "classif.precision", "classif.fbeta", "classif.auc", "classif.tp", "classif.tn", "classif.fp", "classif.fn", "time_both", "time_predict", "time_train"))
N_EVALS <- 30
N_SECS <- 120

learners <- c(
  # ct = lrn("classif.rpart", id = "ct", predict_type = "prob")

  # , ct_tuned = auto_tuner(
  #   tuner = tnr("hyperband", eta = 2, # Reduction factor - in each round, only the top 1/eta of configurations are kept
  #               repetitions = 1),
  #   learner = lrn("classif.rpart", id = "ct", predict_type = "prob"),
  #   resampling = rsmp_cv10, #rsmp_repeated_cv_f5_r2, #rsmp_cv10, #rsmp_cv3, #rsmp_holdout75
  #   measure = msr_mcc,
  #   terminator = trm("run_time", secs = N_SECS),
  #   #terminator = trm("evals", n_evals = N_EVALS),
  #   search_space = lts("classif.rpart.rbv2", #maxdepth = to_tune(p_int(lower = 1, upper = 30, tags = "budget"))
  #                      #minsplit = to_tune(p_int(lower = 1, upper = 64, tags = "budget"))
  #                      cp = to_tune(
  #                        p_dbl(
  #                          lower = 0.0001,
  #                          upper = 0.1,
  #                          tags = "budget"
  #                        )
  #                      )
  #                      )
  # )


  # , lgbm = lrn("classif.lightgbm", id = "lgbm", predict_type = "prob")

  # , lgbm_tuned = auto_tuner(
  #   tuner = tnr("hyperband", eta = 2, # Reduction factor - in each round, only the top 1/eta of configurations are kept
  #               repetitions = 1),
  #   learner = lrn(
  #     "classif.lightgbm"
  #     , id = "lgbm"
  #     , predict_type = "prob"
  #     , boosting = 'dart'
  #     # default=gbdt Use dart, which has been shown to outperform standard GBDTs.
  #     , num_iterations = to_tune(p_int(
  #       lower = 50,
  #       upper = 1500,
  #       tags = "budget"
  #     )),
  #     # default=100 number of boosting iterations. Controls the number of boosting iterations and, therefore, the number of trees built.
  #     learning_rate = to_tune(
  #       lower = 0.01,
  #       upper = 0.75,
  #       logscale = TRUE
  #     ),
  #     min_data_in_leaf = to_tune(lower = 1, upper = 60),
  #     # default=20 minimal number of data in one leaf. Can be used to deal with over-fitting. Larger values reduce overfitting.
  #     num_leaves = to_tune(lower = 10, upper = 300) # default=31 max number of leaves in one tree
  #   ),
  #   resampling = rsmp_cv10,
  #   measure = msr_mcc,
  #   terminator = trm("run_time", secs = N_SECS)
  #   #terminator = trm("evals", n_evals = N_EVALS)
  # )
  # yug,

  # catboost does not handle integers (but handles numeric) in mlr3
  cb = lrn("classif.catboost", id = "cb", predict_type = "prob")

  # , cb_tuned = auto_tuner(
  #   tuner = tnr("hyperband", eta = 2, # Reduction factor - in each round, only the top 1/eta of configurations are kept
  #               repetitions = 1),
  #   #tnr("grid_search", resolution = 5, batch_size = 5),
  #   learner = lrn(
  #     "classif.catboost"
  #     , id = "cb"
  #     , predict_type = "prob"
  #     # https://catboost.ai/docs/en/references/training-parameters/common#learning_rate
  #     # The number of boosting iterations (trees) to be run. It determines the number of trees in the ensemble. You should tune this based on the trade-off between computation time and model performance.
  #     # A higher number of iterations may lead to better performance but requires more computation time.
  #     , iterations = to_tune(p_int(
  #       lower = 50,
  #       upper = 1500,
  #       tags = "budget"
  #     )),
  #     depth = to_tune(p_int(lower = 3, upper = 10)) # default=6
  #   ),
  #   resampling = rsmp_cv10, #rsmp("repeated_cv", folds = 5, repeats = 2), # = rsmp("holdout", ratio = 0.75),
  #   measure = msr_mcc,
  #   terminator = trm("run_time", secs = N_SECS)
  #   #terminator = trm("evals", n_evals = N_EVALS)
  # )

  # ,
  # Random Forest
  #  rf = lrn(
  #   "classif.ranger"
  #   , id = "rf"
  #   , predict_type = "prob"
  #   #, na.action = "na.learn" #default set to "na.learn" to internally handle missing values
  # )

  # , rf_tuned = auto_tuner(
  #   tuner = tnr("hyperband", eta = 2, # Reduction factor - in each round, only the top 1/eta of configurations are kept
  #               repetitions = 1),
  #   learner = lrn("classif.ranger"
  #                 , id = "rf"
  #                 , predict_type = "prob"
  #                 ),
  #   resampling = rsmp_cv10,
  #   measure = msr_mcc,
  #   terminator = trm("run_time", secs = N_SECS),
  #   #terminator = trm("evals", n_evals = N_EVALS),
  #   search_space = lts("classif.ranger.default", num.trees = to_tune(
  #     p_int(
  #       lower = 50,
  #       upper = 1500,
  #       tags = "budget"
  #     )
  #   ))
  # )


  # #Naive Bayes
  # , nb_imp_sample = as_learner(
  #   po("imputesample") %>>%
  #     lrn(
  #     "classif.naive_bayes"
  #     , id = "nb_imp_sample"
  #     , predict_type = "prob"
  #     )
  # )

  # , nb_imp_mode = as_learner(
  #   po("imputemode") %>>%
  #     lrn(
  #       "classif.naive_bayes"
  #       , id = "nb_imp_mode"
  #       , predict_type = "prob"
  #     )
  # )


  # , nb_imp_sample_tuned <- auto_tuner(
  #   learner =
  #     as_learner(
  #       #po("imputehist") %>>%
  #       po("imputesample") %>>%
  #         lrn(
  #           "classif.naive_bayes"
  #           , id = "nb_imp_sample"
  #           , predict_type = "prob" # "response"
  #           , laplace = to_tune(p_dbl(lower = 0.0, upper = 0.1)) #1)) # Smoothing parameter default = 0, [0, +inf]
  #         )),
  #   resampling = rsmp_cv10,
  #   measure = msr("classif.mcc"),
  #   terminator = trm("run_time", secs = N_SECS),
  #   #terminator = trm("evals", n_evals = N_EVALS),
  #   tuner = tnr("random_search")
  # )

  # , nb_imp_mode_tuned <- auto_tuner(
  #   learner =
  #     as_learner(
  #       #po("imputehist") %>>%
  #       po("imputemode") %>>%
  #         lrn(
  #           "classif.naive_bayes"
  #           , id = "nb_imp_mode"
  #           , predict_type = "prob" # "response"
  #           , laplace = to_tune(p_dbl(lower = 0.0, upper = 0.1)) #1)) # Smoothing parameter default = 0, [0, +inf]
  #         )),
  #   resampling = rsmp_cv10, #rsmp("repeated_cv", folds = 5, repeats = 2), #N=10, mcc=0.163, precision=0.163, training_time=81
  #   measure = msr("classif.mcc"),
  #   terminator = trm("run_time", secs = N_SECS),
  #   #terminator = trm("evals", n_evals = N_EVALS),
  #   tuner = tnr("random_search")
  # )
)

# END ----------------------------------------------

print(learners)

# Custom partition function
time_based_partition <- function(task, ratio = 0.67) {
  task <- assert_task(as_task(task, clone = TRUE))
  assertNumeric(ratio, min.len = 1, max.len = 1)

  if (ratio >= 1) {
    stopf("'ratio' must be smaller than 1")
  }

  # Calculate the number of rows for the test set
  n_test <- floor((1 - ratio) * task$nrow)
  # Calculate the number of rows for the training set
  n_train <- task$nrow - n_test

  # Ensure that the test set is the last 'n_test' rows
  train_indices <- 1:n_train
  test_indices <- (n_train + 1):task$nrow
  validation_indices <- integer(0)

  return(
    list(
      train = train_indices,
      test = test_indices,
      validation = validation_indices
    )
  )
}



#---------------------------------------------
tasks <- list()
custom_resamplings <- list()
windows <- c( # expanding windows
  # "1_2", "1_3", "1_4", "1_5", "1_6", #present also in sliding windows
  # "2_3", "2_4", "2_5", "2_6", #present also in sliding windows
  # "3_4", "3_5", "3_6", #present also in sliding windows
  # "4_5", "4_6", #present also in sliding windows
  # "5_6" #present also in sliding windows
  # sliding windows
  "1_2", "2_3", "3_4", "4_5", "5_6",
  "1_3", "2_4", "3_5", "4_6",
  "1_4", "2_5", "3_6",
  "1_5", "2_6",
  "1_6"
  # sliding windows - reversed order
  # "1_6",
  # "1_5", "2_6",
  # "1_4", "2_5", "3_6",
  # "1_3", "2_4", "3_5", "4_6",
  # "1_2", "2_3", "3_4", "4_5", "5_6"
)
for (window in windows) {
  # sliding_window_prediction(window)
  print(paste0("window: ", window))
  task_name <- paste0("task_", window)
  last_char <- substr(task_name, nchar(task_name), nchar(task_name))
  QCdata_name <- paste0("QCdata_", window)
  QCdata <- get(QCdata_name)
  QCdata_test_name <- paste0("QCdata_", last_char)
  QCdata_test <- get(QCdata_test_name)
  task <- TaskClassif$new(task_name, QCdata, "TEST.STATUS")
  split_train_test <- time_based_partition(task, ratio = (1 - nrow(QCdata_test) /
    nrow(QCdata)))

  custom_resampling <- rsmp("custom")
  custom_resampling$instantiate(task,
    train_sets = list(split_train_test$train),
    test_sets = list(split_train_test$test)
  )

  tasks[[task_name]] <- task

  custom_resamplings[[task_name]] <- custom_resampling
}

design <- benchmark_grid(tasks,
  learners,
  custom_resamplings,
  #' * With `paired` set to `TRUE`, tasks and resamplings are treated as pairs.
  #'   I.e., you must provide as many tasks as corresponding instantiated resamplings.
  #'   The grid will be generated based on the Cartesian product of learners and pairs.
  paired = TRUE
)


print(design)
lgr$info("USE_PERSISTED_RESULTS==FALSE - Benchmark models from scratch and save results to files")

lgr$info("Benchmark start")
bmr <- progressr::with_progress(benchmark(design, store_models = FALSE)) # , store_models = TRUE))
print(bmr)

ba <- progressr::with_progress(mlr3benchmark::as_benchmark_aggr(bmr, measures = measures))
print(ba)

# save results to file
if (DATE_DECOMPOSED == TRUE) {
  lgr$info("DATE_DECOMPOSED==TRUE - Save results to files")
  saveRDS(bmr, file = "bmr_DATE_DECOMPOSED.RDS")
  saveRDS(ba, file = "ba_DATE_DECOMPOSED.RDS")
} else {
  lgr$info("DATE_DECOMPOSED==FALSE - Save results to files")
  saveRDS(bmr, file = "bmr.RDS")
  saveRDS(ba, file = "ba.RDS")
}

# Write the tibble to an Excel file
writexl::write_xlsx(
  as_tibble(ba$data),
  "results_walk-forward_validation_with_sliding_and_expanding_window.xlsx"
  # "results_walk-forward_validation_with_sliding_window.xlsx"
)

toc()

# FEATURE IMPORTANCE VIA DALEX

# https://github.com/ModelOriented/DALEX/blob/master/R/misc_loss_functions.R
# Implementation of a new loss function (one minus MCC) designed to work
# with the DALEX package
loss_one_minus_mcc <- function(observed, predicted, cutoff = 0.5, na.rm = TRUE) {
  tp <- sum((observed == 1) * (predicted >= cutoff), na.rm = na.rm)
  fp <- sum((observed == 0) * (predicted >= cutoff), na.rm = na.rm)
  tn <- sum((observed == 0) * (predicted < cutoff), na.rm = na.rm)
  fn <- sum((observed == 1) * (predicted < cutoff), na.rm = na.rm)

  l1 <- as.numeric(tp * tn)
  l2 <- as.numeric(fp * fn)
  m <- sqrt(tp + fp) * sqrt(tp + fn) * sqrt(tn + fp) * sqrt(tn + fn)
  mcc <- as.numeric((l1 - l2) / m)

  1 - mcc
}

get_loss_one_minus_mcc <- function(cutoff = 0.5, na.rm = TRUE) {
  function(o, p) loss_one_minus_mcc(o, p, cutoff = cutoff, na.rm = na.rm)
}
attr(loss_one_minus_mcc, "loss_name") <- "One minus MCC"


# XAI - Feature Importance via DALEX
feat_imp <- function(task, learner = lrn("classif.ranger", predict_type = "prob"), loss_function = loss_one_minus_mcc, N_samples = 1000) {
  data <- task$data() %>% select(-"TEST.STATUS")
  target <- task$data() %>% select("TEST.STATUS")
  # By deafult classification tasks supports only numercical 'y' parameter.
  # Consider changing to numerical vector with 0 and 1 values.
  # QCdata_1$TEST.STATUS <- as.numeric(QCdata_1$TEST.STATUS) - 1 # Passed = 1, Failed = 0
  target$TEST.STATUS <- as.numeric(target$TEST.STATUS) - 1 # Passed = 1, Failed = 0
  # message('target$TEST.STATUS = ', target$TEST.STATUS)

  learner$train(task)

  explainer <- DALEXtra::explain_mlr3(learner,
    data = data,
    y = target,
    label = paste(forstringr::str_extract_part(learner$id, before = FALSE, pattern = "."), task$id)
  )

  perf_QC <- model_performance(explainer)
  perf_QC
  plot(perf_QC)
  plot(perf_QC, geom = "boxplot")
  plot(perf_QC, geom = "histogram")
  plot(perf_QC, geom = "prc")

  # N - number of observations that should be sampled for calculation of variable importance.
  # If NULL then variable importance will be calculated on whole dataset (no sampling)., type = "shap")
  var_importance <- DALEX::model_parts(explainer, loss_function = loss_one_minus_mcc, N = N_samples)
  var_importance

  plot(var_importance)
}


N_SAMPLES <- # 10000 5000 #NULL #round(0.0001*task_all$nrow)
  if (DATE_DECOMPOSED == TRUE) {
    file_name_RF <- "QCdata_ALL_RF_FeatureImportanceViaDALEX_DATE_DECOMPOSED_V3_N10000.pdf"
    file_name_CatBoost <- "QCdata_ALL_CatBoost_FeatureImportanceViaDALEX_DATE_DECOMPOSED_V3_N10000.pdf"
    file_name_ClassTree <- "QCdata_ALL_ClassTree_FeatureImportanceViaDALEX_DATE_DECOMPOSED_V3_N10000.pdf"
  } else {
    file_name_RF <- "QCdata_ALL_RF_FeatureImportanceViaDALEX_V3_N10000.pdf"
    file_name_CatBoost <- "QCdata_ALL_CatBoost_FeatureImportanceViaDALEX_V3_N10000.pdf"
    file_name_ClassTree <- "QCdata_ALL_ClassTree_FeatureImportanceViaDALEX_V3_N10000.pdf"
  }

pdf(file = file_name_RF, paper = "a4r")
ranger_feat_imp_plot <- feat_imp(task = task_all, learner = learners$rf, loss_function = loss_one_minus_mcc, N_samples = N_SAMPLES)
ranger_feat_imp_plot
dev.off.crop(file = file_name_RF)

# pdf(file = file_name_CatBoost, paper = "a4r")
# catboost_feat_imp_plot <- feat_imp(task = task_all, learner = learners$cb, loss_function = loss_one_minus_mcc, N_samples = N_SAMPLES)
# catboost_feat_imp_plot
# dev.off.crop(file = file_name_CatBoost)

# pdf(file = file_name_ClassTree, paper = "a4r")
# classtree_feat_imp_plot <- feat_imp(task = task_all, learner = learners$ct, loss_function = loss_one_minus_mcc, N_samples = N_SAMPLES)
# classtree_feat_imp_plot
# dev.off.crop(file = file_name_ClassTree)
