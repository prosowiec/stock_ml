import streamlit as st
import os
import scraper
import database
import main


option = st.sidebar.selectbox("Select operation", ('Scrape gainers and lossers', 'Contiue scraping from recent symbol file', 
                                                   'Scrape ONLY price from prev scrapes', 'Show / upload data'), 0)

st.header(option)
db = database.Stockdata()


if option == 'Scrape gainers and lossers':
    col1, col2 = st.columns(2)
    with col1:
        st.warning('Warning!! Data in recent.csv will be replaced!', icon = "⚠️")
    with col2:
        agree = st.checkbox('I agree')
        
    if st.button('Start scraping') and agree: 
            os.remove("recent.csv")
            st.write('Process has began')
            s = scraper.Symbols()
            symbols_filename = 'recent_symbols.csv'
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
        symbols_filename = 'recent_symbols.csv'
        s.save_all_sybmols(symbols_filename)
        main.start_scraping(symbols_filename, min_sleep = 10, max_sleep = 20)   
        db.upload_df()    

    st.balloons()
    
if option == 'Show / upload data':
    if st.button('Upload recent df'): 
        db.upload_df()
        
    df = db.import_df_whole()
    st.dataframe(df)
