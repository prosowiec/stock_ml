from tqdm import tqdm
import pandas as pd
import scraper
import time
import random
from stqdm import stqdm

def scrape_param(symbol):
    scrape = scraper.Features(symbol)
    scrape.save_to_csv()

def start_scraping(symbols_filename : str = 'recent_symbols.csv', min_sleep : int = 9, max_sleep : int = 17):
    s = scraper.Symbols()
    headers = ['symbols', 'scraped']
    symbols_df = pd.read_csv(symbols_filename)
    for i in stqdm(symbols_df.index):
        if symbols_df['scraped'][i] == 0:
            try:
                scrape_param(str(symbols_df['symbols'][i]))
                symbols_df.loc[i, 'scraped'] = 1  
                time.sleep(random.randint(min_sleep, max_sleep))
            except:
                continue
        symbols_df.to_csv(symbols_filename, index = False, mode = 'w', header = headers)


if __name__ == '__main__':
    s = scraper.Symbols()
    symbols_filename = 'recent_symbols.csv'
    s.save_all_sybmols(symbols_filename)
    start_scraping(symbols_filename)



