#  Feature engineering and data preparation


* Sample

    I used data starting from date_id = 700, as this is when the number of time_ids stabilizes at 968
* Data preparation

    Simple standardization and NaN imputation with zero were applied

* Feature engeneering
     1. Market averages: Averages per date_id and time_id.
     2. Rolling statistics: Rolling averages and standard deviations over the last 1000 time_ids for each symbol.


# Base model 
  Time-series GRU with sequence equal to one day. I ended up with two slightly different architectures:


1.  3-layer GRU
2.  1-layer GRU followed by 2 linear layers with ReLU activation and dropout.

# Ensemble

I ran both models on 3 seeds and took a simple unweighted average of predictions from those 6 models
