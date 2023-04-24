import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import xgboost as xgb

import database

def split_data(df):
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('price_after')
    test_labels = test_features.pop('price_after')
    
    return train_features, test_features, train_labels, test_labels

def real_up_dows(row):
    if row['price_after'] >= row['before']:
        return 1
    else:
        return 0

def pred_up_dows(row):
    if row['price_after'] >= row['pred']:
        return 1
    else:
        return 0

def classify_pred(before, test_labels, test_predictions):
    dataset = pd.DataFrame({'price_after': test_labels, 'pred': test_predictions}, columns=['price_after', 'pred'])
    dataset['before'] = before
    dataset['real_up_dows'] = dataset.apply(lambda row: real_up_dows(row), axis=1)
    dataset['pred_up_dows'] = dataset.apply(lambda row: pred_up_dows(row), axis=1)
    dataset.loc[dataset['real_up_dows'] == dataset['pred_up_dows'], 'check_pred'] = 1
    dataset.loc[dataset['real_up_dows'] != dataset['pred_up_dows'], 'check_pred'] = 0
    
    return dataset

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(15, activation='relu'),
        layers.Dense(15, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model

def prepare_df(df):
    df = df.loc[df['price_after'] != 0]
    df.pop('_id')
    df.pop('symbol')
    df.tail()
    #convert row to NaN hen format is not correct
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

    return df

def train_reg_tensorflow(df = ''):
    df = prepare_df(df)
    train_features, test_features, train_labels, test_labels = split_data(df)
    
    normalizer = tf.keras.layers.Normalization(axis=-1)
    train_features = tf.convert_to_tensor(train_features, dtype=tf.float32)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    test_features = tf.convert_to_tensor(test_features, dtype=tf.float32)
    test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

    normalizer.adapt(np.asarray(train_features).astype(np.float32))

    dnn_model = build_and_compile_model(normalizer)
    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=200)

    test_predictions = dnn_model.predict(test_features).flatten()
    dataset = classify_pred(test_features[:, 0], test_labels, test_predictions)
    accuracy  = round(dataset['check_pred'].sum() / dataset.shape[0] * 100, 2)

    pred = list(dataset['pred_up_dows'])
    test = list(dataset['real_up_dows'])
    fpr, tpr, _ = metrics.roc_curve(test,  pred)
    auc = metrics.roc_auc_score(test, pred)
    mape = metrics.mean_absolute_percentage_error(test_features[:, 0], test_predictions)

    return accuracy, history, fpr, tpr, auc, dnn_model, mape


def train_reg_xgboost(df = ''):

    df = prepare_df(df)
    train_features, test_features, train_labels, test_labels = split_data(df)
    
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=1000,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=6,
                       learning_rate=0.01,
                       )
    
    reg.fit(train_features, train_labels,
        eval_set=[(train_features, train_labels), (test_features, test_labels)],
        eval_metric = 'mape',
        verbose=100)
    
    test_predictions = reg.predict(test_features)
    dataset = classify_pred(test_features['current_price'], test_labels, test_predictions)
    accuracy  = round(dataset['check_pred'].sum() / dataset.shape[0] * 100, 2)

    pred = list(dataset['pred_up_dows'])
    test = list(dataset['real_up_dows'])
    fpr, tpr, _ = metrics.roc_curve(test,  pred)
    auc = metrics.roc_auc_score(test, pred)
    mape = metrics.mean_absolute_percentage_error(test_features['current_price'], test_predictions)
    return accuracy, fpr, tpr, auc, reg, mape

def dnn_tensor_predict(model, cur_features):
    symbols = cur_features.pop('symbol')
    pred_dnn = model.predict(cur_features).flatten()
    df = pd.DataFrame({'symbol': symbols, 'scraped_price' : cur_features['current_price'],'predicted_price_dnn': pred_dnn}, 
                      columns=['symbol', 'scraped_price', 'predicted_price_dnn'])
    return df

def xgboost_predict(model, cur_features):
    symbols = cur_features.pop('symbol')
    pred_xgb = model.predict(cur_features)
    df = pd.DataFrame({'symbol': symbols, 'scraped_price' : cur_features['current_price'], 'predicted_price_xgb': pred_xgb}, 
                      columns=['symbol', 'scraped_price', 'predicted_price_xgb'])
    return df

def pred_up_dows_prod(row, pred_name):
    if row['scraped_price'] >= row[pred_name]:
        return 1
    else:
        return 0

def same_movement(dnn_reg, xgb_reg):
    dnn_reg['up_dows_dnn'] = dnn_reg.apply(lambda row: pred_up_dows_prod(row, 'predicted_price_dnn'), axis=1)
    xgb_reg['up_dows_xgb'] = xgb_reg.apply(lambda row: pred_up_dows_prod(row, 'predicted_price_xgb'), axis=1)
    xgb_reg = xgb_reg.drop('scraped_price', axis=1)
    df = pd.concat([dnn_reg.set_index('symbol'), xgb_reg.set_index('symbol')], axis=1)
    df = df.loc[df['up_dows_xgb'] == df['up_dows_dnn']]
    df = df.drop(['up_dows_xgb'], axis=1)
    
    return df

