from bs4 import BeautifulSoup
import requests
import regex as re
from multiprocessing.pool import ThreadPool
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

class Symbols:
    def __init__(self, gainers_url : str = 'https://finance.yahoo.com/gainers?offset=0&count=100', 
                 losers_url : str = 'https://finance.yahoo.com/losers?offset=0&count=100'):
        self.gainers_url = gainers_url
        self.losers_url = losers_url
        self.headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        self.losers_symbols = []
        self.gainers_symbols = []
        
    
    def get_symbols(self, url) -> list:
        r = requests.get(url, headers = self.headers, timeout=1000)
        soup = BeautifulSoup(r.content, "lxml")
        symbols_table = soup.find_all("table", {"class":"W(100%)"})[0]
        symbols_pattern = re.compile('(?<=data-symbol=\")(.*?)(?=\")')
        symbols = set(re.findall(symbols_pattern, str(symbols_table)))
        return list(symbols)
    
    def get_symbols_gainers(self) -> list:
        if not self.gainers_symbols:
            self.gainers_symbols = self.get_symbols(self.gainers_url)
        return self.gainers_symbols
    
    def get_symbols_losers(self) -> list:
        if not self.losers_symbols:
            self.losers_symbols = self.get_symbols(self.losers_url)
        return self.losers_symbols
    

class Features:
    def __init__(self, symbol:str):
        self.symbol = symbol
        self.current_price = self.week_change = self.half_year_change = self.year_change = ''
        self.soup_summary = ''
        self.soup_history = ''
        self.soup_financials = ''
        self.headers = {"User-Agent" : 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        
            
    def make_many_soups(self):
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        dt = timedelta(days=366)
        datefrom = int((now - dt).timestamp())
        
        pool = ThreadPool()
        self.soup_summary = pool.apply_async(self.make_soup, (f'https://finance.yahoo.com/quote/{self.symbol}?p={self.symbol}',)).get()
        self.soup_financials = pool.apply_async(self.make_soup, (f'https://finance.yahoo.com/quote/{self.symbol}/financials?p={self.symbol}',)).get()
        self.soup_cashflow = pool.apply_async(self.make_soup, (f'https://finance.yahoo.com/quote/{self.symbol}/cash-flow?p={self.symbol}',)).get()
        self.soup_history = pool.apply_async(self.make_soup, (f'https://finance.yahoo.com/quote/{self.symbol}/history?period1={datefrom}&period2={dateto}&interval=1wk&filter=history&frequency=1wk&includeAdjustedClose=true',)).get()
    
    def make_soup(self, url):
        r = requests.get(url, headers = self.headers, timeout=1000)
        soup = BeautifulSoup(r.content, "lxml")
        return soup     
     
    #summary page
    def get_1d_stock_change(self):
        param_table = self.soup_summary.find_all("div", {"id":"smartDaConfig"})[0]
        pattern_change = re.compile('(?<=\"FIN_TICKER_PRICE_CHANGE_PERCENT\":\")(.*?)(?=%\")')
        symbol_change = re.findall(pattern_change, str(param_table))[0]
        return float(symbol_change)
    
    def get_eps(self):
        eps_ratio = self.soup_summary.find_all("td", {"data-test":"EPS_RATIO-value"})[0]
        if 'N/A' in eps_ratio:
            return 0.0
        else:
            return float(eps_ratio.text)
    
    def get_market_cap(self):
        market_cap = str(self.soup_summary.find_all("td", {"data-test":"MARKET_CAP-value"})[0].text)
        if 'B' in market_cap:
            market_cap = float(market_cap.replace('B',''))
        elif 'T' in market_cap:
            market_cap = round(float(market_cap.replace('T',''))*100, 2)
        return market_cap
        
    def get_pe_ratio(self):
        pe_ratio = self.soup_summary.find_all("td", {"data-test":"PE_RATIO-value"})[0].text
        if pe_ratio != 'N/A':
            return float(pe_ratio)
        #N/A case
        return -1.0             
    
    #financials page       
    def get_3y_revenue_change(self):
        revenue_change = []
        try:
            revenue_quarterly = self.soup_financials.find_all("div", {"D(tbr) fi-row Bgc($hoverBgColor):h"})
            revenue_symbol = re.compile('(?<=data-test=\"fin-col\"><span>)(.*?)(?=<\/span><\/div>)')
            revenue_list = re.findall(revenue_symbol, str(revenue_quarterly[0]))
        except IndexError:
            return 0,0,0
            
        if revenue_list[0] == revenue_list[1]:
            revenue_list.pop(0)
        i = 0
        for rev in reversed(revenue_list):
            rev = int(rev.replace(',',''))
            if i == 0:
                i += 1
                temp = rev
                continue
            else:
                revenue_change.append(round((rev-temp)/temp,2)*100)
            temp = rev
        revenue_change = revenue_change[::-1]

        if len(revenue_change) < 3:
            while len(revenue_change) != 3:
                revenue_change.append(0)
        elif len(revenue_change) > 3:
            while len(revenue_change) != 3:
                revenue_change.pop()
                
        return revenue_change[0], revenue_change[1], revenue_change[2]
    
    def get_free_cash_flow_3y(self):
        free_cach_flow_change = []
        
        try:
            free_cash = self.soup_cashflow.find_all("div", {"class" : "D(tbr) fi-row Bgc($hoverBgColor):h"})
            cash_symbol = re.compile('(?<=data-test=\"fin-col\"><span>)(.*?)(?=<\/span><\/div>)')
            cash_list = re.findall(cash_symbol, str(free_cash[-1]))
        except IndexError:
            return 0,0,0
        
        if cash_list[0] == cash_list[1]:
            cash_list.pop(0)
        i = 0
        for flow in reversed(cash_list):
            flow = float(flow.replace(',',''))
            if i == 0:
                i += 1
                temp = flow
                continue
            else:
                free_cach_flow_change.append(round(((flow-temp)*100)/temp ,2))
            temp = flow
        free_cach_flow_change = free_cach_flow_change[::-1]

        if len(free_cach_flow_change) < 3:
            while len(free_cach_flow_change) != 3:
                free_cach_flow_change.append(0)
        elif len(free_cach_flow_change) > 3:
            while len(free_cach_flow_change) != 3:
                free_cach_flow_change.pop()
       
        return free_cach_flow_change[0], free_cach_flow_change[1], free_cach_flow_change[2]
    
    def historical_price(self):
        closed_prece_list = []
        price_history = self.soup_history.find_all("tr", {"class" : "BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)"})
        for row in price_history:
                price_symbol = re.compile('(?<=<span>)(.*?)(?=<\/span>)')
                price_list = re.findall(price_symbol, str(row))
                
                #split case 
                if len(price_list) == 2:
                        continue
                closed_prece_list.append(float(price_list[-3]))
                
        self.current_price = closed_prece_list[0]
        if len(closed_prece_list) < 2:
            self.week_change = 0
        else:
            self.week_change = round(((self.current_price - closed_prece_list[2])* 100) / closed_prece_list[2], 2)
        if len(closed_prece_list) < 26:
            self.half_year_change = 0
            self.year_change = 0
        else:
            self.half_year_change = round(((self.current_price - closed_prece_list[26])* 100) / closed_prece_list[26], 2)
        self.year_change = round(((self.current_price - closed_prece_list[-1])* 100) / closed_prece_list[-1], 2)
        
    def get_cur_price(self):
        if not self.current_price:
            self.historical_price()
        return self.current_price
   
    def get_week_change(self):
        if not self.current_price:
            self.historical_price()
        return self.week_change
    
    def get_half_year_change(self):
        if not self.current_price:
            self.historical_price()
        return self.half_year_change
    
    def get_year_change(self):
        if not self.current_price:
            self.historical_price()
        return self.year_change
    
    def save_to_csv(self, filename : str = 'test.csv'):
        headers = ['symbol', 'current_price', 'pe_ratio', 'eps_ratio', 'market_cap', 'day_change', 'week_change', 'half_year_change', 
                   'year_change', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y', 
                   'revenue_flow_change_1y', 'revenue_change_2y', 'revenue_change_3y']
        
        self.make_many_soups()
        pool = ThreadPool()
        r1, r2, r3 = zip(pool.apply_async(self.get_free_cash_flow_3y, ()).get())
        fc1, fc2, fc3 = zip(pool.apply_async(self.get_free_cash_flow_3y, ()).get())
        
        data = {
            'symbol' : self.symbol,
            'current_price' : pool.apply_async(self.get_cur_price, ()).get(),
            'pe_ratio' : pool.apply_async(self.get_pe_ratio, ()).get(), 
            'eps_ratio' : pool.apply_async(self.get_eps, ()).get(), 
            'market_cap' : pool.apply_async(self.get_market_cap, ()).get(), 
            'day_change' : pool.apply_async(self.get_1d_stock_change, ()).get(), 
            'week_change' : pool.apply_async(self.get_week_change, ()).get(), 
            'half_year_change' : pool.apply_async(self.get_half_year_change, ()).get(), 
            'year_change' : pool.apply_async(self.get_year_change, ()).get(), 
            'free_cach_flow_change_1y' : fc1, 
            'free_cach_flow_change_2y' : fc2, 
            'free_cach_flow_change_3y' : fc3, 
            'revenue_change_1y' : r1, 
            'revenue_change_2y' : r2, 
            'revenue_change_3y' : r3,
        }
        df = pd.DataFrame(data=data)
        df.to_csv(filename, index = False, mode='a', header=False)
        
s = Symbols()
symbol_l = s.get_symbols_losers()
print(symbol_l)
for symbol in tqdm(symbol_l):
    test = Features(symbol)
    filename = 'test.csv'
    test.save_to_csv(filename)
    