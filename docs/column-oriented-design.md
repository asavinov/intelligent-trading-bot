# Column-oriented design for feature engineering and machine learning

## What is column-orientation?

Column-oriented is a view on how data is processed and how the processing steps are described.
The main idea is that data is represented and processed as columns belonging to tables.
This is opposed to representing and processing data as rows belonging to tables.
In other words, the column-oriented approach treats columns as first-class elements of the data processing pipeline.

In column-oriented approach, a new column is defined in terms of other columns.
Mathematically, this means defining a new function (a mapping from one set to another set) in terms of other functions.
For comparison, in the relational data model, a new collection (mathematical set) of rows is defined in terms of other collections.
In spreadsheets (like Excel), a new cell value is defined in terms of other cells.

In the general case, columns belong to different tables and we need
operations for both defining columns (functions) and populating tables (sets). However, for the purposes
of this project, it is enough to have one table which is populated manually by appending rows and all
operations with data are reduced to defining columns.
For example, if we want to define a new column C which is a sum of two existing columns A and B, then we
could express this as a formula: C=A+B.

## Why column-orientation?

There are several reasons why this approach is suitable for time series analysis and trading.

One benefit is that the approach based on defining columns (features) in terms of already existing columns is very simple and natural.
Once a feature has beed defined, it can be used in defining other features and so on.
As a result, data processing is represented as a directed acyclic graph (DAG) of feature definitions.
It is quite easy to maintain this graph by incrementally adding new features,
adjusting already existing features and validating intermediate results. The nodes of this graph are
well modularized units which are evaluated independently and can be re-used for different purposes.
Thus this approach is good basis for feature engineering which is known to be a crucial step in time series analysis
and forecasting. In many cases, it is the quality of the chosen features that mainly influences the quality of the final forecast.

The column-oriented approach can be also applied to how new labels used for training machine learning models are defined.
In many cases, the labels (what we want to predict) are not future values. Instead, we want to have some more aggregated
information, for example, reachability of certain criteria or identifying certain events.
The difference of labels from normal features is that labels are generated from future data while
features are generated from past (and current) data. Apart from that, they are defined in the same way as
columns depending on the values in other columns.

It is possible to define some kind of script language for defining new features.
Its expressions will be either interpreted and translated into the target language (Python in our case).
However, in this project we use *user-defined functions* which programmatically define how the values of the output column
are computed from the values of the input columns.

## Unifying feature engineering and machine learning

Another advantage which is specific to this project is that column-orientation allows us to unite feature engineering and machine learning.

Any feature can be and normally should be parameterized. These parameters can be set found and set manually.
It is precisely how traditional features are defined which represent a piece of expert knowledge.
Another way to define feature parameters is to extract from historic data. It is how machine learning works.
We process large amount of historic data like stock quotes, apply some machine learning algorithm, and this
algorithm find optimal values of paramters from the data. Frequently we even do not know what are these values
because for us it is important only that they will minimize certain loss function which represents what we want to predict.
Such a feature will use these optimal automatically found parameters to compute its output values (predictions)
instead of manually set parameters.

What is new in this project is that it unifies traditional features which specify exactly how to compute output values
with trainable features which know how to extract its optimal parameters from historic data.
Such features are referred to as *trainable features*.
Such trainable feature is a combination of its computational (prediction) part and its trainable (learn) part.
Normally they are based on some machine learning or statistical algorithm.

## A sequence of analysis

Data analysis consists of the following three steps:
- Data input. Here it is necessary to retrieve data from external sources and set the values of the source columns.
The source columns are not computed but rather their values are set from outside.
Note that normally only a small number of missing records is retrieved, these records are appended and
after that the source columns have valid values
- Feature evaluation. Here all feature definitions are evaluated and the values of all derived columns are computed.
Only missing values which have not be computed before should be computed (for performance reasons).
After this step, all derived columns have valid values. Note that these columns include also those predicted by ML-algorithms.
- Data output. This steps uses the latest data in derived columns in order to send them to external consuming services.
This step may include sending notifications with visualizations, performing trades, adujsting existing orders or writing
results in a database for further analysis.
