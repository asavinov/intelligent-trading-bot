# Scripts

Scripts are used for retrieving data, computing derived features, training ML models, hyper-parameter search, training signal models.

Most scripts rely on the `App` class and its configuration parameters.

## Download historic (klines) data

Execute: `python -m scripts.download_data`

Notes:
* Edit main in binance-data.py by setting necessary symbol and function arguments
* Script can be started from local folder and will store result in this folder.
* The script will check if the previous file in the current folder and append it with new data. If not found, then new file is created and all data will be retrieved
* Currently we use 2 separate sources stored in 2 separate files:
  * klines (spot market), its history is quite long
  * futures (also klines but from futures market), its history is relative short

## Merge historic data into one dataset

Execute: `python -m scripts.merge_data`

Notes:
* Set source file paths in the script
* Merge historic data into one dataset. We analyse data using one common time raster and different column names. Also, we fix problems with gaps by producing a uniform time raster. Note that columns from different source files will have different history length so short files will have Nones in the long file.
* If necessary, edit input data file locations as absolute paths

## Generate feature matrix

Execute: `python -m scripts.generate_features`

Here we compute derived features (also using window-functions) and produce a feature matrix. We also compute target features (labels). Note that we compute many possible features and labels but not all of them have to be used. In parameters, we define past history length (windows) and future horizon for labels. Currently, we generate 3 kinds of features independently: klines features (source 1), future features (source 2), and label features (our possible predictions targets).

Notes:
* Check and set input and output directories and file names
* Ensure that latest source data has been downloaded from binance server (previous step)
* The goal s to load source (kline) data, generate derived features and labels, and store the result in output file. The output is supposed to be used for other procedures like training prediction models.
* Max past window and max future horizon are currently not used (None will be stored)
* Future horizon for labels is hard-coded. Change if necessary
* If necessary, uncomment line with storing to parquet (install the packages)
* Output file will store features and labels as they are implemented in the trade module
* Same number of lines in output as in input file
* Approximate time: ~20-30 minutes

## Generate rolling predictions

Execute: `python -m scripts.generate_rolling_predictions`

Generate rolling predictions. Here we train a model using previous data less frequently, say, once per day or week, but use much more previous data than in typical window-based features. We apply then one constant model to predict values for the future time until it is re-trained again using the newest data. (If the re-train frequency is equal to sample rate, that is, we do it for each new row, then we get normal window-based derived feature with large window sizes.) Each feature is based on some algorithm with some hyper-parameters and some history length. This procedure does not choose best hyper-parameters - for that purpose we need some other procedure, which will optimize the predicted values with the real ones. Normally, the target values of these features are directly related to what we really want to predict, that is, to some label. Output of this procedure is same file (feature matrix) with additional predicted features (scores). This file however will be much shorter because we need some quite long history for some features (say, 1 year). Note that for applying rolling predictions, we have to know hyper-parameters which can be found by a simpler procedure.

Notes:
* Prerequisite: We already have to know the best prediction model(s) and its best parameters
* There can be several models used for rolling predictions
* Essentially, the predicting models are treated here as (more complex) feature definitions
* Choose the best model and its parameters using grid search (below)
* The results of this step are consumed by signal generator and backtesting

## (Grid) search for best parameters of and/or best prediction models

Execute: `python -m scripts.grid_search`

The goal is to find the best prediction models, and their best parameters using hyper-parameter optimization. The results of this step are certain model (like nn, gradient boosting etc.) and, importantly, its best hyper-parameters.

Notes:
* The results are consumed by the rolling prediction step
* There can be many algorithms and many historic horizons or input feature set

## Train prediction models

Execute: `python -m scripts.train_predict_models` 

Here we regularly train prediction models to be used in the production service as parameters of the corresponding predicted feature generation procedures.

Notes:
* There can be many predicted features and models, for example, for spot and future markets or based on different prediction algorithms or historic horizons
* The procedure will consume feature matrix and hence the following files should be updated: source data, merge files, generate features (no need to generate rolling features).
* The generated models have to be copied to the folder where they are found by the signal/trade server

## Train signal models

Here we find the best parameters for signal generation like thresholds.

Execute: `python -m scripts.train_signal_models`

The input is a feature matrix with all scores (predicted features). Our goal is to define a feature the output of which will be directly used for buy/sell decisions. We need search for the best hyper-parameters starting from simple score threshold and ending with some data mining algorithm.

Notes:
* We consume the results of rolling predictions
* We assume that rolling prediction produce many highly informative features
* The grid search (brute force) of this step has to test our trading strategy using back testing as (direct) metric. In other words, trading performance on historic data is our metric for brute force or simple ML 
* Normally the result is some thresholds or some simple ML model
* Important: The results of this step are consumed in the production service to generate signals 
