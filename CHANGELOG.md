# Change Log

* v0.7.dev
  * make server and the scripts work with different frequencies rather than only 1 minute
  * restructure config for ML-features making their configuration closer to normal features and normal generators
  * possibility to reference arbitrary external functions as generators  
  * support for parquet storage format for intermediate files
  * all data including features, predicted scores and signals are stored in the context and available for further processing
    * Improved visualization of historic data in on-line mode by using this common data context
  * refactor and improve train signals based on the new structure
  * introduce a section with signal generators
  * refactoring: implement aggregations and trade logic to conventional column generators

* v0.6.0 (2023-10-05)
  * add visualization of previous transactions
  * move algorithm configurations from source code to configuration file algorithms section
  * add a feature generator based on TA-lib technical analysis library
  * add configurations to feature generators instead of global parameters
  * add configuration parameters to label generators
  * refactor aggregation logic

* v0.5.0 (2023-03-26)
  * refactor and improve signal training (grid search) by taking into account the new rule model and external config file 
  * introduce rules with multiple inputs (currently only two dimensional rules with two thresholds)
  * introduce a new rule model with different rule type 
  * introduce multiple aggregation sets
  * generalize the aggregation model by separating it from the rule model, refactoring the logic of score combination and other improvements
  * implement a new labeling function

* v0.4.0 (2022-09-04)
  * Unified specifications of data_sources, feature_sets and label_sets
  * Introduce buy labels and sell labels
  * Logging trade transactions and computing performance of the service
  * Support for daily data frequency and Yahoo data source (only in batch mode) 

* v0.3.0 (2022-07-09)

* v0.2.0 (2022-03-19)

* v0.1.0 (2021-11-01)

* v0.0.0 (2020-02-23)
  * initial commit
