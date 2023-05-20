import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


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
    tf.random.set_seed(2137)
    
    dnn_model = build_and_compile_model(normalizer)
    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose = 0, epochs=200)

    test_predictions = dnn_model.predict(test_features).flatten()
    dataset = classify_pred(test_features[:, 0], test_labels, test_predictions.copy())
    accuracy  = round(dataset['check_pred'].sum() / dataset.shape[0] * 100, 2)

    pred = list(dataset['pred_up_dows'])
    test = list(dataset['real_up_dows'])
    fpr, tpr, _ = metrics.roc_curve(test,  pred)
    auc = metrics.roc_auc_score(test, pred)
    mape = metrics.mean_absolute_percentage_error(test_features[:, 0], test_predictions)

    return accuracy, history, fpr, tpr, auc, dnn_model, mape


def train_reg_xgboost(df = '', tune = True):

    df = prepare_df(df)
    train_features, test_features, train_labels, test_labels = split_data(df)
    
    if tune:
        space = xgb_baysian(train_features, train_labels, test_features, test_labels)
    else:
        space = {'colsample_bytree': 1.5454067387890569, 'early_stopping_rounds': 170.0, 'gamma': 0.3030738510672555, 'learning_rate': 0.063, 
                 'max_depth': 8.0, 'min_child_weight': 2.0, 'n_estimators': 1400.0, 'reg_alpha': 0.0, 'reg_lambda': 0.9607086474969695}

    reg = xgb.XGBRegressor(
                        n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']), objective='reg:squarederror', booster='gbtree', eval_metric = "mape",
                        learning_rate = space['learning_rate'], early_stopping_rounds = int(space['early_stopping_rounds']))
    
    reg.fit(train_features, train_labels,
        eval_set=[(train_features, train_labels), (test_features, test_labels)],
        verbose = 0)
    
    test_predictions = reg.predict(test_features)
    dataset = classify_pred(test_features['current_price'], test_labels, test_predictions.copy())
    accuracy  = round(dataset['check_pred'].sum() / dataset.shape[0] * 100, 2)
    pred = list(dataset['pred_up_dows'])
    test = list(dataset['real_up_dows'])
    
    fpr, tpr, _ = metrics.roc_curve(test,  pred)
    auc = metrics.roc_auc_score(test, pred)
    mape = metrics.mean_absolute_percentage_error(test_features['current_price'], test_predictions)
    return accuracy, fpr, tpr, auc, reg, mape

def make_quantity_df(df):
    df['pos_procendage'] = (df['scraped_price'] * 100) / df['scraped_price'].sum()
    df['quantity'] = 1 / df['pos_procendage']
    df['quantity'] = df['quantity'].round(0)

    def new_column(row):
        if row['quantity'] == 0:
            return 1
        else:
            return row['quantity']

    df['quantity'] = df.apply(new_column, axis=1)
    
    return df

def dnn_tensor_predict(model, cur_features):
    cur_features.loc[:, 'current_price':] = cur_features.loc[:, 'current_price':].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna() 
    symbols = cur_features.pop('symbol')
    
    pred_dnn = model.predict(cur_features).flatten()
    df = pd.DataFrame({'symbol': symbols, 'scraped_price' : cur_features['current_price'],'predicted_price_dnn': pred_dnn}, 
                      columns=['symbol', 'scraped_price', 'predicted_price_dnn'])
    df['order_type'] = df.apply(lambda row: pred_ORDER_prod(row, 'predicted_price_dnn'), axis=1)
    df = make_quantity_df(df)
    return df


def xgboost_predict(model, cur_features):
    cur_features.loc[:, 'current_price':] = cur_features.loc[:, 'current_price':].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna() 
    symbols = cur_features.pop('symbol')

    pred_xgb = model.predict(cur_features)

    df = pd.DataFrame({'symbol': symbols, 'scraped_price' : cur_features['current_price'], 'predicted_price_xgb': pred_xgb}, 
                      columns=['symbol', 'scraped_price', 'predicted_price_xgb'])
    df['order_type'] = df.apply(lambda row: pred_ORDER_prod(row, 'predicted_price_xgb'), axis=1)
    return df

def pred_ORDER_prod(row, pred_name):
    if row['scraped_price'] >= row[pred_name]:
        return 'SELL'
    else:
        return 'BUY'

def pred_up_dows_prod(row, pred_name):
    if row['scraped_price'] >= row[pred_name]:
        return 1
    else:
        return 0


def same_movement(dnn_reg, xgb_reg):
    dnn_reg['up_dows_dnn'] = dnn_reg.apply(lambda row: pred_up_dows_prod(row, 'predicted_price_dnn'), axis=1)
    xgb_reg['up_dows_xgb'] = xgb_reg.apply(lambda row: pred_up_dows_prod(row, 'predicted_price_xgb'), axis=1)
    xgb_reg = xgb_reg.drop('scraped_price', axis=1)
    df = pd.concat([dnn_reg.set_index('symbol'), xgb_reg.set_index('symbol')], axis = 1)
    df = df.loc[df['up_dows_xgb'] == df['up_dows_dnn']]
    df = df.drop(['up_dows_dnn'], axis=1)
    
    return df

def eval_combined_df(dnn_model, xgb_model, df):
    df.pop('_id')
    train_features, test_features, train_labels, test_labels = split_data(df)
    test_predictions_tensorflow = dnn_tensor_predict(dnn_model, test_features.copy())
    test_predictions_xgboost = xgboost_predict(xgb_model, test_features.copy())
    
    test_predictions_xgboost['real_price'] = test_labels
    
    eval = same_movement(test_predictions_tensorflow, test_predictions_xgboost)
    eval['up_dows_real'] = eval.apply(lambda row: pred_up_dows_prod(row, 'real_price'), axis=1)
    eval['symbol'] = eval.index
    
    auc = metrics.roc_auc_score(eval['up_dows_xgb'], eval['up_dows_real'])
    accuracy = metrics.accuracy_score(eval['up_dows_xgb'], eval['up_dows_real'])
    fpr, tpr, _ = metrics.roc_curve(eval['up_dows_xgb'], eval['up_dows_real'])

    return accuracy, auc, fpr, tpr


def xgb_baysian(X_train, y_train, X_test, y_test):
    space = {'max_depth': hp.quniform("max_depth", 3, 9, 1),
        'gamma': hp.uniform('gamma', 0, 3),
        'reg_alpha' : hp.quniform('reg_alpha', 0, 5, 1),
        'reg_lambda' : hp.uniform('reg_lambda', 0, 2),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0, 2),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 5, 1),
        'n_estimators': hp.quniform("n_estimators", 1000, 1500, 100),
        'learning_rate': hp.quniform("learning_rate", 0.001, 0.3, 0.001),
        'seed': 2137,
        'early_stopping_rounds' : hp.quniform("early_stopping_rounds", 50, 200, 10)
    }

    def objective(space):
        oreg = xgb.XGBRegressor(
                        n_estimators = int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']), objective='reg:squarederror', booster='gbtree', eval_metric = "rmse",
                        learning_rate = space['learning_rate'], early_stopping_rounds = int(space['early_stopping_rounds']))
        
        evaluation = [( X_train, y_train), ( X_test, y_test)]
        
        oreg.fit(X_train, y_train,
                eval_set=evaluation,
                verbose = False)
        
        pred = oreg.predict(X_test)
        
        accuracy = metrics.mean_absolute_percentage_error(y_test, pred)
        return {'loss': accuracy, 'status': STATUS_OK }
    
    trials = Trials()
    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 100,
                            trials = trials)
    
    return best_hyperparams


def get_metrics(test_features, test_labels, test_predictions):
    dataset = classify_pred(test_features, test_labels, test_predictions.copy())

    dataset['pred_change'] = (dataset['pred'] - dataset['before']) * 100 / dataset['before']
    #dataset = dataset.loc[(abs(dataset['pred_change']) > 0.1) &  (abs(dataset['pred_change']) < 5)]
    
    accuracy  = round(dataset['check_pred'].sum() / dataset.shape[0] * 100, 2)
    pred = list(dataset['pred_up_dows'])
    test = list(dataset['real_up_dows'])
    
    fpr, tpr, _ = metrics.roc_curve(test,  pred)
    auc = metrics.roc_auc_score(test, pred)
    mape = metrics.mean_absolute_percentage_error(test_features, test_predictions)
    return accuracy, fpr, tpr, auc, mape

    
def eval_models(dnn_model, xgb_model, df):
    df_b = df.copy()
    df = prepare_df(df)
    train_features, test_features, train_labels, test_labels = split_data(df)
    
    test_predictions_tensorflow = dnn_model.predict(test_features.copy()).flatten()
    test_predictions_xgboost = xgb_model.predict(test_features.copy())
    accuracy_ten, fpr_ten, tpr_ten, auc_ten, mape_ten = get_metrics(test_features['current_price'], test_labels, test_predictions_tensorflow)
    accuracy_xgb, fpr_xgb, tpr_xgb, auc_xgb, mape_xgb = get_metrics(test_features['current_price'], test_labels, test_predictions_xgboost)
    accuraccy_both, auc_both, both_fpr, both_tpr = eval_combined_df(dnn_model, xgb_model, df_b)

    res = pd.DataFrame({'model' : ['tensorflow', 'xgb', 'both'], 
                        'accuracy' : [accuracy_ten, accuracy_xgb, accuraccy_both],
                        'mape' : [mape_ten, mape_xgb, min(mape_ten, mape_xgb)],
                        'auc' : [auc_ten, auc_xgb, auc_both],
                        'fpr' : [fpr_ten, fpr_xgb, both_fpr],
                        'tpr' : [tpr_ten, tpr_xgb, both_tpr],
                        }, 
                        columns=['model', 'accuracy','mape', 'auc', 'fpr', 'tpr'])

    return res