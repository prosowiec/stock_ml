>### Work in progress !!! ###
Application scrapes most volatile(1D) equities od given day from Yahoo Finance. Set up in the way, that locally only stores symbols and data from current or previous day, therefore most of data is in NoSQL database - MongoDB. 

My strategy is based upon classifying market moves, that aren't revelat to company performence in the past. I want to quantify them and by preforming basic machine learning regressions choose movement of given stock(up/down) and predict profit of transaction if I would execute it. 

I' am still working on implementing xgboost and tensorflow regression algorithms. For to day's hyperparameters models accurasy of around ~80%, after binary classifying whether price went up or down on given prediction.
