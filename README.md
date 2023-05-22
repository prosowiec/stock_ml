Application scrapes most volatile(1D) equities od given day from Yahoo Finance. Set up in the way, that locally only stores symbols and data from current or previous day, therefore most of data is in NoSQL database - MongoDB. 

My strategy is based upon classifying market moves, that aren't revelat to company performence in the past. I want to quantify them and by preforming basic machine learning regressions choose movement of given stock(up/down) and predict profit of transaction if I would execute it. 

For to day's hyperparameters models accuracy stand on around ~80%, after binary classifying whether price went up or down on given prediction. Hyperparameters in XGBOOST bodel can be optimased using Bayesian optimization, but the same can not be done in Tensorflow model, because after tunning model did on average was losing accuracy, cause of limited amount of data. 

Orders made by models can by executed in dashboard using Interactive Brokers API.
