import pandas as pd
from pymongo import MongoClient, UpdateOne
import os

class Stockdata:
    def __init__(self):
        self.client =  MongoClient(os.environ.get('con_string'))
        self.db = self.client[os.environ.get("stockdata")]
        self.collection = self.db['stonks']
        self.df = ''
        
    def get_df(self, filename):
        if self.df:
            return self.df
        
        headers = ['symbol', 'current_price', 'pe_ratio', 'eps_ratio', 'market_cap', 'day_change', 'week_change', 'half_year_change', 
            'year_change', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y', 
            'revenue_change_1y', 'revenue_change_2y', 'revenue_change_3y', 'price_after']
        self.df = pd.read_csv('test.csv', names = headers)
        
        return self.df
    
    def upload_df(self):
        data = self.get_df('test2.csv')
        data_dict = data.to_dict("records")
        self.collection.insert_many(data_dict)
    
    def import_df(self):
        df = pd.DataFrame(list(self.collection.find()))
        return df
        
    def upload_current_price(self, df):
        updates = []

        
        for _, row in df.iterrows():
            updates.append(UpdateOne({'_id': row.get('_id')}, {'$set': {'price_after': row.get('price_after')}}, upsert=True))

        self.db.stonks.bulk_write(updates)
