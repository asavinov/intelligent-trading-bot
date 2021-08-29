# Data/knowledge processing pipeline

## Main steps

Note: see also start.py

#### 1. Download historic (klines) data

* Download historic data. Currently we use 2 separate sources stored in 2 separate files:
  * klines (spot market), its history is quite long
  * futures (also klines but from futures market), its history is relative short

* Script: `scripts.download_data.py` or `python start.py download_data`

* Script can be started from local folder and will store result in this folder.

Notes:
* Edit main in binance-data.py by setting necessary symbol
* Run script binance-data.py which will directly call get_klines_all()
* Get klines for futures: Use the same function get_klines_all() but uncomment the section at the beginning.

#### 2. Merge historic data into one dataset

* Merge historic data into one dataset. We analyse data using one common raster and different column names. Also, we fix problems with gaps by producing a uniform time raster. Note that columns from different source files will have different history length so short file will have Nones in the long file.

* Script: `scripts.merge_data.py`

* If necessary, edit input data file locations as absolute paths

#### 3. Generate feature matrix

* Generate features. Here we compute derived features (also using window-functions) and produce a final feature matrix. We also compute target features (labels). Note that we compute many possible features and labels but not all of them have to be used. In parameters, we define past history length (windows) and future horizon for labels. Currently, we generate 3 kinds of features independently: klines features (source 1), future features (source 2), and label features (our possible predictions targets).

* Script: `scripts.generate_features.py` or `python start.py generate_features`

Notes:
* The goal here is to load source (kline) data, generate derived features and labels, and store the result in output file. The output is supposed to be used for other procedures like training prediction models.
* Ensure that latest source data has been downloaded from binance server (previous step)
* Max past window and max future horizon are currently not used (None will be stored)
* Future horizon for labels is hard-coded (currently 300). Change if necessary
* If necessary, uncomment line with storing to parquet (install the packages)
* Output file will store features and labels as they are implemented in the trade module. Copy the header line to get the list.
* Same number of lines in output as in input file
* Approximate time: ~20-30 minutes (on YOGA)

#### 4. Generate rolling predictions

* Generate rolling predictions. Here we train a model using previous data less frequently, say, once per day or week, but use much more previous data than in typical window-based features. We apply then one constant model to predict values for the future time until it is re-trained again using the newest data. (If the re-train frequency is equal to sample rate, that is, we do it for each new row, then we get normal window-based derived feature with large window sizes.) Each feature is based on some algorithm with some hyper-parameters and some history length. This procedure does not choose best hyper-parameters - for that purpose we need some other procedure, which will optimize the predicted values with the real ones. Normally, the target values of these features are directly related to what we really want to predict, that is, to some label. Output of this procedure is same file (feature matrix) with additional predicted features (scores). This file however will be much shorter because we need some quite long history for some features (say, 1 year). Note that for applying rolling predictions, we have to know hyper-parameters which can be found by a simpler procedure.

* Script: `scripts.generate_rolling_predictions.py` or `python start.py generate_rolling_predictions.py`

Notes:
* Prerequisite: We already have to know the best prediction model(s) and its best parameters
* There can be several models used for rolling predictions
* Essentially, the predicting models are treated here as (more complex) feature definitions
* Choose the best model and its parameters using grid search (below)
* The results of this step are consumed by signal generator

#### 5. (Grid) search for best parameters of and/or best prediction models

The goal is to find the best prediction models, and their best parameters using hyper-parameter optimization. The results of this step are certain model (like nn, gradient boosting etc.) and, importantly, its best hyper-parameters.

Notes:
* The results are consumed by the rolling prediction step
* There can be many algorithms and many historic horizons or input feature set

* Script: `grid_search.py`

#### 6. Train prediction models

Here we regularly train prediction models to be used in the production service as parameters of the corresponding predicted feature generation procedures.

Notes:
* There can be many predicted features and models, for example, for spot and future markets or based on different prediction algorithms or historic horizons
* The procedure will consume feature matrix and hence the following files should be updated: source data, merge files, generate features (no need to generate rolling features).
* The generated models have to be copied to the folder where they are found by the signal/trade server

Script: `train_predict_models.py` 

#### 7. Train signal models

Here we find the best parameters for signal generation like thresholds.

* Train signal models. The input is a feature matrix with all scores (predicted features). Our goal is to define a feature the output of which will be directly used for buy/sell decisions. We need search for the best hyper-parameters starting from simple score threshold and ending with some data mining algorithm.

* Script: `scripts.train_signal_models.py` or `python start.py train_signal_models.py`

Notes:
* We consume the results of rolling predictions
* We assume that rolling prediction produce many highly informative features
* The grid search (brute force) of this step has to test our trading strategy using back testing as (direct) metric. In other words, trading performance on historic data is our metric for brute force or simple ML 
* Normally the result is some thresholds or some simple ML model
* Important: The results of this step are consumed in the production service to generate signals 
