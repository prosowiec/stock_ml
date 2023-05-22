import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
import plotly.express as px
import os
import scraper
import database
import main
import train_ml
import algo_trade

option = st.sidebar.selectbox("Select operation", ('Scrape gainers and lossers', 'Contiue scraping from recent symbol file', 
                                                   'Scrape ONLY price from prev scrapes', 'Predict and save moves', 'Train ML prediction models',
                                                   'Evaluate models performance', 'Place orders', 'Show / upload data'), 0)
st.sidebar.write('Created by Łukasz Janikowski')

db = database.Stockdata()
st.header(option)

if option == 'Scrape gainers and lossers':
    col1, col2 = st.columns(2)
    with col1:
        st.warning('Warning!! Data in recent.csv will be replaced!', icon = "⚠️")
    with col2:
        agree = st.checkbox('I agree')
        
    if st.button('Start scraping') and agree: 
            try:
                os.remove("operations/recent.csv")
            except:
                pass
            st.write('Process has began')
            s = scraper.Symbols()
            symbols_filename = 'operations/recent_symbols.csv'
            s.save_all_sybmols(symbols_filename)
            main.start_scraping(symbols_filename, min_sleep = 10, max_sleep = 20)   
            db.upload_df()    

            st.balloons()
            
if option == 'Scrape ONLY price from prev scrapes':
    st.write('Click button to continue')
    if st.button('Start scraping'): 
        with st.spinner('Please wait ~~ aprox. 15 min'):
            c1 = scraper.Features()
            data = db.import_df_0_price()
            df = c1.get_df_price_after(data)
            db.upload_current_price(df)    
    
if option == 'Contiue scraping from recent symbol file':
    st.write('Process has began')
    if st.button('Start scraping'):
        s = scraper.Symbols()
        symbols_filename = 'operations/recent_symbols.csv'
        main.start_scraping(symbols_filename, min_sleep = 10, max_sleep = 20)   
        db.upload_df()    

    st.balloons()
    
    
if option == 'Show / upload data':
    if st.button('Upload recent df'): 
        db.upload_df()
        
    df = db.import_df_whole()
    st.dataframe(df)


if option == 'Train ML prediction models':
    col1, col2 = st.columns(2)
    df = db.import_df_whole()
    headers = ['symbol', 'current_price', 'pe_ratio', 'eps_ratio', 'market_cap', 'day_change', 'week_change', 'half_year_change', 
            'year_change', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y', 
            'revenue_change_1y', 'revenue_change_2y', 'revenue_change_3y', 'price_after']
    
    xg_reg, ten_reg = None, None
    agree = st.checkbox('Perform hypertuning parameters in xgboost')
    if st.button('Start training models'):
        with col1:
            st.write('XGBOOST MODEL')
            with st.spinner('Wait for training to finish'):
                xg_accuracy, xg_fpr, xg_tpr, xg_auc, xg_reg, xg_mape = train_ml.train_reg_xgboost(df, agree)
            
            st.write(f'{round(xg_mape, 2)} MAPE')
            st.write(f'{round(xg_auc, 2)} AUC')
            chart_data = pd.DataFrame(xg_tpr, xg_fpr)
            st.area_chart(chart_data)
            fi = pd.DataFrame(data=xg_reg.feature_importances_[1:],
            index = xg_reg.feature_names_in_[1:], columns=['importance'])
            st.write('Feature Importance')
            fig = px.bar(fi.sort_values('importance'), orientation='h')
            st.plotly_chart(fig)
            
            xg_reg.save_model("saved_models/xg_reg.json")

        with col2:
            st.write('TENSORFLOW MODEL')
            with st.spinner('Wait for training to finish'):
                ten_accuracy, ten_history, ten_fpr, ten_tpr, ten_auc, ten_reg, ten_mape = train_ml.train_reg_tensorflow(df)
            
            st.write(f'{round(ten_mape, 2)} MAPE')
            st.write(f'{round(ten_accuracy, 2)} AUC')
            chart_data = pd.DataFrame(ten_tpr, ten_fpr)
            st.area_chart(chart_data)
            
            fig, ax = plt.subplots(figsize=(10,10))
            ax2=ax.twinx()
            line1 = ax.plot(ten_history.history['loss'], label='loss', color = "tab:blue")
            line2 = ax2.plot(ten_history.history['val_loss'], label='val_loss', color = "orange")
            ax.legend(loc=0)
            ax2.legend(loc=1)
            st.write('History chart')
            st.pyplot(fig)
            
            ten_reg.save('saved_models/dnn_reg')
        
        accuraccy_both, auc_both, both_fpr, both_tpr = train_ml.eval_combined_df(ten_reg, xg_reg, df)
        st.write(f'Accuraccy of model with same movement - {round(accuraccy_both, 2)}')
        chart_data_both = pd.DataFrame(both_tpr, both_fpr)
        st.area_chart(chart_data)
          
        performace_df = pd.DataFrame({'model' : ['tensorflow', 'xgb', 'both'], 
                            'accuracy' : [ten_accuracy, xg_accuracy, accuraccy_both],
                            'mape' : [ten_mape, xg_mape, min(ten_mape, xg_mape)],
                            'auc' : [ten_auc, xg_auc, auc_both],
                            'fpr' : [ten_fpr, xg_fpr, both_fpr],
                            'tpr' : [ten_tpr, xg_tpr, both_tpr],
                            }, 
                            columns=['model', 'accuracy','mape', 'auc', 'fpr', 'tpr'])
        
        performace_df.to_csv('saved_models/performace.csv', index = False, mode='w')
        

if option == 'Predict and save moves':
    dnn_model = tf.keras.models.load_model(f'saved_models/dnn_reg')
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model("saved_models/xg_reg.json")
    data = db.import_df_0_price()
    try:
        data = data.drop('price_after', axis=1)
        data = data.drop('_id', axis=1)
        per_df = pd.read_csv('saved_models/performace.csv')
        col1, col2 = st.columns(2)
        with col1:
            st.write('XGBOOST MODEL')
            st.write(F'MAPE {round(per_df.loc[1][2], 2)}')
            st.write(F'AUC {round(per_df.loc[1][3], 2)}')
            pred_xgb = train_ml.xgboost_predict(xgb_reg, data.copy())
            pred_xgb = train_ml.make_quantity_df(pred_xgb)
            st.dataframe(pred_xgb)
            pred_xgb.to_csv('orders/xgb.csv', index = False, mode='w')

        with col2:
            st.write('TENSORFLOW MODEL')
            st.write(F'MAPE {round(per_df.loc[0][2], 2)}')
            st.write(F'AUC {round(per_df.loc[0][3], 2)}')
            dnn_df = train_ml.dnn_tensor_predict(dnn_model, data.copy())
            dnn_df = train_ml.make_quantity_df(dnn_df)
            st.dataframe(dnn_df)
            dnn_df.to_csv('orders/tensorflow.csv', index = False, mode='w')


        st.write('Symbols with same movement')
        st.write(F'MAPE(best of models) {round(per_df.loc[2][1], 2)}')
        st.write(F'AUC {round(per_df.loc[2][2], 2)}')
        st.dataframe(train_ml.same_movement(dnn_df, pred_xgb))
    except:
        st.warning('There are no prices to be predicted')


if option == 'Evaluate models performance':
    dnn_model = tf.keras.models.load_model(f'saved_models/dnn_reg')
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model("saved_models/xg_reg.json")
    data = db.import_df_whole()
    if st.button('Start evaluation'):
        df = train_ml.eval_models(dnn_model, xgb_reg, data)
        col1, col2 = st.columns(2)
        with col1:
            accuracy, fpr, tpr, auc, mape = df.loc[1, 'accuracy'], df.loc[1, 'fpr'], df.loc[1, 'tpr'], df.loc[1, 'auc'], df.loc[1, 'mape']
            st.write('XGBOOST MODEL')
            st.write(f'{round(mape, 2)} MAPE')
            st.write(f'{round(auc, 2)} AUC')
            st.write(f'{round(accuracy, 2)} ACCURACY')
            chart_data = pd.DataFrame(tpr, fpr)
            st.area_chart(chart_data)

        with col2:
            accuracy, fpr, tpr, auc, mape = df.loc[0, 'accuracy'], df.loc[0, 'fpr'], df.loc[0, 'tpr'], df.loc[0, 'auc'], df.loc[0, 'mape']
            st.write('TENSORFLOW MODEL')           
            st.write(f'{round(mape, 2)} MAPE')
            st.write(f'{round(accuracy, 2)} AUC')
            st.write(f'{round(accuracy, 2)} ACCURACY')
            chart_data = pd.DataFrame(tpr, fpr)
            st.area_chart(chart_data)

        st.write(df)    
        df.to_csv('saved_models/performace.csv', index = False, mode='w')


if option == 'Place orders':
    if st.button('Orders from tensorflow'):
        df = pd.read_csv('orders/tensorflow.csv')
        algo_trade.make_trades(df, 'predicted_price_dnn')
        
    if st.button('Orders from xgboost'):
        df = pd.read_csv('orders/xgb.csv')
        algo_trade.make_trades(df, 'predicted_price_xgb')