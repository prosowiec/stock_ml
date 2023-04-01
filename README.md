### Work in progress; readme may contain my notes ###
1. Pobiera historyczne info SPY - symbol, p/e, eps, markercap
	1.1 Pobiera codzienne info SPY
2. Analizuje dane historyczne sprzed np 3 lat - model regresji
3. Wykrywa 10 największych zmian cen
4. Daje je do RNN, gdzie model wypluwa 10 najbardziej prawdopodobnych pozycji, że pójdą w górę i dół
	4.1. Dodaje wyniki z pop dnia do modelu i uczy na nowo ?po 30 dniach?




UI - streamlit
ML - tensorflow
Dane - ;))

https://finance.yahoo.com/gainers
https://finance.yahoo.com/losers


features
p/e, eps, revenue change(anual), 1y revenue %, 2y revenue %,3y revenue %, 1 year change%, 1 month change%,markercap - to normalize!!

Wczytywanie danych z następnego dnia i zapisywanie w odcielnym pliku do późniejszego fine tune

checking state of financials
https://finance.yahoo.com/quote/{SYMBOL}/cash-flow?p={SYMBOL}
https://finance.yahoo.com/quote/{SYMBOL}/financials?p={SYMBOL}
revenue change(anual), 1y revenue %, 2y revenue %,3y revenue %

Current standing
https://finance.yahoo.com/quote/{SYMBOL}?p={SYMBOL}
p/e, eps, markercap, 1 day change%

Short term stock preformace
https://finance.yahoo.com/quote/{SYMbol}/history?period1=1522281600&period2=1680048000&interval=1mo&filter=history&frequency=1mo&includeAdjustedClose=true
1 week change ,30 days change%, 180 days change , 1 year change

HEADERS | mongodb datastock db -> reupload with future_price scraped after 1d

'current_price', 'pe_ratio', 'eps_ratio', 'market_cap', 'day_change', 'week_change', 'half_year_change', 'year_change', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y'
headers = ['symbol', 'current_price', 'pe_ratio', 'eps_ratio', 'market_cap', 'day_change', 'week_change', 'half_year_change', 
            'year_change', 'free_cach_flow_change_1y', 'free_cach_flow_change_2y', 'free_cach_flow_change_3y', 
            'revenue_change_1y', 'revenue_change_2y', 'revenue_change_3y', 'price_after']