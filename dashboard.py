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

option = st.sidebar.selectbox("Select operation", ('Scrape gainers and lossers', 'Contiue scraping from recent symbol file', 
                                                   'Scrape ONLY price from prev scrapes', 'Predict prices', 'Train ML prediction models',
                                                   'Show / upload data'), 0)

db = database.Stockdata()
st.header(option)

if option == 'Scrape gainers and lossers':
    col1, col2 = st.columns(2)
    with col1:
        st.warning('Warning!! Data in recent.csv will be replaced!', icon = "⚠️")
    with col2:
        agree = st.checkbox('I agree')
        
    if st.button('Start scraping') and agree: 
            os.remove("operations/recent.csv")
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
    with st.spinner('Please wait ~~ aprox. 1h'):
        s = scraper.Symbols()
        symbols_filename = 'operations/recent_symbols.csv'
        s.save_all_sybmols(symbols_filename)
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
            
            chart_data = pd.DataFrame(xg_tpr, xg_fpr)
            xg_reg.save_model("saved_models/xg_reg.json")
            st.write(f'{round(xg_mape, 2)} MAPE')
            st.write(f'{round(xg_auc, 2)} AUC')
            st.area_chart(chart_data)
            fi = pd.DataFrame(data=xg_reg.feature_importances_[1:],
            index = xg_reg.feature_names_in_[1:], columns=['importance'])
            st.write('Feature Importance')
            fig = px.bar(fi.sort_values('importance'), orientation='h')
            st.plotly_chart(fig)

        with col2:
            st.write('TENSORFLOW MODEL')
            with st.spinner('Wait for training to finish'):
                ten_accuracy, ten_history, ten_fpr, ten_tpr, ten_auc, ten_reg, ten_mape = train_ml.train_reg_tensorflow(df)
            chart_data = pd.DataFrame(ten_tpr, ten_fpr)
            st.write(f'{round(ten_mape, 2)} MAPE')
            st.write(f'{round(ten_accuracy, 2)} AUC')
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
        
        auc_both, both_fpr, both_tpr = train_ml.eval_combined_df(ten_reg, xg_reg, df)
        st.write(f'Accuraccy of model with same movement - {round(auc_both, 2)}')
        chart_data_both = pd.DataFrame(both_tpr, both_fpr)
        st.area_chart(chart_data)
          
        performace_df = pd.DataFrame({'model': ['Tensorflow', 'XGBOOST', 'BOTH'], 
                                      'MAPE' : [round(ten_mape, 2), round(xg_mape, 2), max(round(ten_mape, 2), round(xg_mape, 2))], 
                                      'AUC': [round(ten_auc, 2), round(xg_auc, 2), round(auc_both, 2)]}, 
                                     columns=['model', 'MAPE', 'AUC']) 
        
        
        performace_df.to_csv('saved_models/performace.csv', index = False, mode='w')
        

        
if option == 'Predict prices':
    dnn_model = tf.keras.models.load_model(f'saved_models/dnn_reg')
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.load_model("saved_models/xg_reg.json")
    data = db.import_df_0_price()
    data = data.drop('price_after', axis=1)
    data = data.drop('_id', axis=1)
    
    per_df = pd.read_csv('saved_models/performace.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('XGBOOST MODEL')
        st.write(F'MAPE {per_df.loc[1][1]}')
        st.write(F'AUC {per_df.loc[1][2]}')
        pred_xgb = train_ml.xgboost_predict(xgb_reg, data.copy())
        st.dataframe(pred_xgb)

    with col2:
        st.write('TENSORFLOW MODEL')
        st.write(F'MAPE {per_df.loc[0][1]}')
        st.write(F'AUC {per_df.loc[0][2]}')
        dnn_df = train_ml.dnn_tensor_predict(dnn_model, data.copy())
        st.dataframe(dnn_df)

    st.write('Symbols with same movement')
    st.write(F'MAPE(best of models) {per_df.loc[2][1]}')
    st.write(F'AUC {per_df.loc[2][2]}')
    st.dataframe(train_ml.same_movement(dnn_df, pred_xgb))
    
