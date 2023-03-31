import streamlit as st
import scraper


#scraper = scraper.Symbols()
magic = scraper.Features('GOOG')
st.write(magic.get_eps())
magic.save_to_csv()
