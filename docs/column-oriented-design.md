# Column-oriented design for feature engineering and machine learning

## What is column-orientation?

Column-orientation is a perspective on how data is processed and how processing steps are described. The core idea is that data is represented and processed as columns belonging to tables, rather than as rows. In other words, the column-oriented approach treats columns as first-class elements of the data processing pipeline.

In this approach, a new column is defined in terms of existing columns. Mathematically, this means defining a new function (a mapping from one set to another) in terms of other functions. By comparison, in the relational data model, a new collection (mathematical set) of rows is defined in terms of other collections, where a row is identified by its unique combination of attributes. Similarly, in spreadsheets like Excel, a new cell value is defined in terms of other cells using two-dimensional coordinates.

Generally, columns may belong to different tables, requiring operations for both defining columns (functions) and populating tables (sets). However, for the purposes of this project, it is sufficient to use a single table that is populated manually by appending rows; all data operations are then reduced to defining columns. For example, if we want to define a new column `C` as the sum of two existing columns `A` and `B`, we can express this simply as the formula: `C = A + B`.

An example of a column-oriented approach to deriving new data from existing data is shown in the figure below.

![Column-orientations](images/column-orientation.svg)

Here, the table contains source columns (in green) whose values are set explicitly when new rows are appended. Once source values are established, derived columns (in blue) can be evaluated. Each derived value is computed separately for each row. Any derived value can depend on values in the current row as well as previous rows, utilizing both source columns and previously evaluated derived columns. In this example, the first derived column, `span`, computes the difference between `high` and `low` prices within the same row. For the last row, this equals 7 (334 - 327). The second derived column, `span_MA_3`, calculates the moving average of the previously computed `span` values. For the last row, this equals 10.3—the average of the three most recent `span` values: (7 + 13 + 11) / 3.

## Why column-orientation?

There are several reasons why this approach is particularly suitable for time series analysis and trading.

First, defining features in terms of existing columns is intuitive and straightforward. Once a feature is defined, it can serve as a building block for subsequent features. Consequently, data processing is represented as a directed acyclic graph (DAG) of feature definitions. This graph is easy to maintain: new features can be added incrementally, existing ones adjusted, and intermediate results validated independently. The nodes of this graph are modular units that can be reused for different purposes. Thus, this approach provides a solid foundation for feature engineering, which is known to be a crucial step in time series analysis and forecasting. Often, the quality of the chosen features has a greater impact on forecast accuracy than the choice of model itself.

The column-oriented approach also applies to defining labels for training machine learning models. Frequently, labels represent aggregated information—such as the reachability of certain criteria or the occurrence of specific events—rather than simple future values. While labels are typically generated from future data and features from past or current data, they are otherwise defined identically: as columns dependent on values in other columns.

While it is possible to design a domain-specific scripting language for defining features (which would then be interpreted or translated into Python), this project instead uses *user-defined functions*. These functions programmatically define how output column values are computed from input column values, offering maximum flexibility without introducing a new language.

## Unifying feature engineering and machine learning

A key advantage specific to this project is that column-orientation enables the unification of feature engineering and machine learning.

Any feature can—and often should—be parameterized. Parameters can be determined manually, which is how traditional features encode expert knowledge. Alternatively, parameters can be extracted from historical data, which is the essence of machine learning. By processing large volumes of historical data (e.g., stock quotes), an algorithm finds optimal parameter values that minimize a loss function representing the prediction target. The resulting feature uses these automatically optimized parameters to compute outputs instead of relying on manual configuration.

This project unifies traditional features (with explicitly defined computation logic) and trainable features (which learn optimal parameters from data). We refer to these collectively as *trainable features*. A trainable feature combines a computational (prediction) component with a training (learning) component, typically based on machine learning or statistical algorithms.

## A sequence of analysis

Data analysis consists of the following three steps:

-   **Data input.** Data is retrieved from external sources to populate source columns. Unlike derived columns, source columns are not computed; their values are set externally. Typically, only new or missing records are retrieved and appended, ensuring source columns contain valid values.
-   **Feature evaluation.** All feature definitions are evaluated to compute derived column values. For performance reasons, only values that have not been previously computed are calculated. After this step, all derived columns—including those generated by ML algorithms—contain valid values.
-   **Data output.** The latest derived column values are sent to external consuming services. This step may include sending notifications with visualizations, executing trades, adjusting existing orders, or writing results to a database for further analysis.

## Links to related projects

-   [Prosto](https://github.com/asavinov/prosto): Prosto is a data processing toolkit radically changing how data is processed by heavily relying on functions and operations with functions-an alternative to map-reduce and join-groupby
-   [Lambdo](https://github.com/asavinov/lambdo): Feature engineering and machine learning: together at last!
