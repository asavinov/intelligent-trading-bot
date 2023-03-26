# Change Log

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
