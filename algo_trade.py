from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order_condition import Create, OrderCondition
from ibapi.order import *
from stqdm import stqdm
import threading
import time
import websocket
import json
from datetime import datetime

class IBapi(EWrapper, EClient):
	def __init__(self):
		EClient.__init__(self, self)
		self.contract_details = {} #Contract details will be stored here using reqId as a dictionary key

	def nextValidId(self, orderId: int):
		super().nextValidId(orderId)
		self.nextorderId = orderId
		print('The next valid order id is: ', self.nextorderId)

	def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
		print('orderStatus - orderid:', orderId, 'status:', status, 'filled', filled, 'remaining', remaining, 'lastFillPrice', lastFillPrice)
	
	def openOrder(self, orderId, contract, order, orderState):
		print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action, order.orderType, order.totalQuantity, orderState.status)

	def execDetails(self, reqId, contract, execution):
		print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)

def run_loop(app):
	app.run()

def Stock_contract(symbol, secType='STK', exchange='SMART', currency='USD'):
	''' custom function to create stock contract '''
	contract = Contract()
	contract.symbol = symbol
	contract.secType = secType
	contract.exchange = exchange
	contract.currency = currency
	return contract

def order_buy(symbol, app, lmt_buy, stop_loss, quantity = 1):
    contract = Stock_contract(symbol)
    order = Order()
    order.action = 'BUY'
    order.totalQuantity = quantity
    order.orderType = 'LMT'
    order.lmtPrice = lmt_buy
    order.orderId = app.nextorderId
    app.nextorderId += 1
    order.transmit = False
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    
    #Create stop loss order object
    stop_order = Order()
    stop_order.action = 'SELL'
    stop_order.totalQuantity = quantity
    stop_order.orderType = 'STP'
    stop_order.auxPrice = stop_loss
    stop_order.orderId = app.nextorderId
    app.nextorderId += 1
    stop_order.parentId = order.orderId    
    stop_order.eTradeOnly = False
    stop_order.firmQuoteOnly = False
    
    order.transmit = True
    #Place orders
    app.placeOrder(order.orderId, contract, order)
    app.placeOrder(stop_order.orderId, contract, stop_order)
    
    
def order_sell(symbol, app, lmt_buy, stop_loss, quantity = 1):
    contract = Stock_contract(symbol)
    order = Order()
    order.action = 'SELL'
    order.totalQuantity = quantity
    order.orderType = 'LMT'
    order.lmtPrice = lmt_buy
    order.orderId = app.nextorderId
    app.nextorderId += 1
    order.transmit = False
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    
    #Create stop loss order object
    stop_order = Order()
    stop_order.action = 'BUY'
    stop_order.totalQuantity = quantity
    stop_order.orderType = 'STP'
    stop_order.auxPrice = stop_loss
    stop_order.orderId = app.nextorderId
    app.nextorderId += 1
    stop_order.parentId = order.orderId    
    stop_order.eTradeOnly = False
    stop_order.firmQuoteOnly = False
    
    order.transmit = True
    #Place orders
    app.placeOrder(order.orderId, contract, order)
    app.placeOrder(stop_order.orderId, contract, stop_order)


def make_trades_IB(df, pred_coln_name):
    app = IBapi()
    app.connect('127.0.0.1', 7497, 123)
    app.nextorderId = None
    api_thread = threading.Thread(target=run_loop, args=(app, ), daemon=True)
    api_thread.start()

    T = 0
    while True and T != 5:
        if isinstance(app.nextorderId, int):
            print('connected')
            break
        else:
            print('waiting for connection')
            T += 1
            time.sleep(1)
        if T == 5:
            print('Connection error')
            return 0

    for index, row in stqdm(df.iterrows()):
        symbol = row['symbol']
        lmt_buy = row['scraped_price']
        stop_loss = row[pred_coln_name]
        quantity = row['quantity']
        if row['order_type'] == 'BUY':
            order_buy(symbol, app, lmt_buy, stop_loss, quantity)
        else:
            order_sell(symbol, app, lmt_buy, stop_loss, quantity)
        time.sleep(1 / 5)
        
        
################ XTB ####################
        
# https://github.com/Saitama298/Python-XTB-API/blob/main/API.py
class XTB:

    def __init__(self, ID, PSW):
        self.ID = ID
        self.PSW = PSW
        self.ws = 0
        self.streamSessionId = 0
        self.exec_start = self.get_time()
        self.connect()
        self.login()    
        
    def login(self):
        login ={
            "command": "login",
            "arguments": {
                "userId": self.ID,
                "password": self.PSW
            }
        }
        login_json = json.dumps(login)
        result = self.send(login_json)
        result = json.loads(result)
        status = result["status"]
        self.streamSessionId = result["streamSessionId"]
        if str(status)=="True":
            return True
        else:
            return False

    def logout(self):
        logout ={
            "command": "logout"
        }
        logout_json = json.dumps(logout)
        result = self.send(logout_json)
        result = json.loads(result)
        status = result["status"]
        self.disconnect()
        if str(status)=="True":
            return True
        else:
            return False


    def get_ServerTime(self):
        time ={
            "command": "getServerTime"
        }
        time_json = json.dumps(time)
        result = self.send(time_json)
        result = json.loads(result)
        time = result["returnData"]["time"]
        return time

    def get_Balance(self):
        balance ={
            "command": "getMarginLevel"
        }
        balance_json = json.dumps(balance)
        result = self.send(balance_json)
        result = json.loads(result)
        balance = result["returnData"]["balance"]
        return balance


    def get_Symbol(self, symbol):
        symbol ={
            "command": "getSymbol",
            "arguments": {
                "symbol": symbol
            }
        }
        symbol_json = json.dumps(symbol)
        result = self.send(symbol_json)
        result = json.loads(result)
        symbol = result["returnData"]
        return symbol
    
    def get_tick_prices(self, symbol, level = 0):
        symbol = {
            "command": "getTickPrices",
            "arguments": {
                "level": level,
                "symbols": [symbol],
                "timestamp": time.time()
            }
        }
        symbol_json = json.dumps(symbol)
        result = self.send(symbol_json)
        result = json.loads(result)
        try:
            symbol = result["returnData"]
        except:
            return result
        return result
    
    def make_Trade(self, symbol, price, cmd, transaction_type, volume, comment="", order=0, sl=0, tp=0, days=0, hours=0, minutes=0):

        delay = self.to_milliseconds(days=days, hours=hours, minutes=minutes)
        if delay==0:
            expiration = self.get_ServerTime() + self.to_milliseconds(minutes=1)
        else:
            expiration = self.get_ServerTime() + delay
        
        TRADE_TRANS_INFO = {
            "cmd": cmd,
            "customComment": comment,
            "expiration": expiration,
            "offset": -1,
            "order": order,
            "price": price,
            "sl": sl,
            "symbol": symbol,
            "tp": tp,
            "type": transaction_type,
            "volume": volume
        }
        trade = {
            "command": "tradeTransaction",
            "arguments": {
                "tradeTransInfo": TRADE_TRANS_INFO
            }
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        if result["status"]==True:
            #success
            return True, result["returnData"]["order"]
        else:
            #error
            return False, 0
        """
        format TRADE_TRANS_INFO:
        cmd	        Number	            Operation code
        customComment	String	            The value the customer may provide in order to retrieve it later.
        expiration	Time	            Pending order expiration time
        offset	        Number	            Trailing offset
        order	        Number	            0 or position number for closing/modifications
        price	        Floating number	    Trade price
        sl	        Floating number	    Stop loss
        symbol	        String	            Trade symbol
        tp	        Floating number	    Take profit
        type	        Number	            Trade transaction type
        volume	        Floating number	    Trade volume

        values cmd:
        BUY	        0	buy
        SELL	        1	sell
        BUY_LIMIT	2	buy limit
        SELL_LIMIT	3	sell limit
        BUY_STOP	4	buy stop
        SELL_STOP	5	sell stop
        BALANCE	        6	Read only. Used in getTradesHistory  for manager's deposit/withdrawal operations (profit>0 for deposit, profit<0 for withdrawal).
        CREDIT	        7	Read only

        values transaction_type:
        OPEN	    0	    order open, used for opening orders
        PENDING	    1	    order pending, only used in the streaming getTrades  command
        CLOSE	    2	    order close
        MODIFY	    3	    order modify, only used in the tradeTransaction  command
        DELETE	    4	    order delete, only used in the tradeTransaction  command
        """

    def check_Trade(self, order):
        trade ={
            "command": "tradeTransactionStatus",
            "arguments": {
                    "order": order
            }
        }
        trade_json = json.dumps(trade)
        result = self.send(trade_json)
        result = json.loads(result)
        status = result["returnData"]["requestStatus"]
        return status
    '''
    ERROR 	0	error
    PENDING	1	pending
    ACCEPTED	3	The transaction has been executed successfully
    REJECTED	4	The transaction has been rejected
    '''

    def get_History(self, start=0, end=0, days=0, hours=0, minutes=0):
        if start!=0:
            start = self.time_conversion(start)
        if end!=0:
            end = self.time_conversion(end)

        if days!=0 or hours!=0 or minutes!=0:
            if end==0:
                end = self.get_ServerTime()
            start = end - self.to_milliseconds(days=days, hours=hours, minutes=minutes)
        
        history ={
            "command": "getTradesHistory",
            "arguments": {
                    "end": end,
                    "start": start
            }
        }
        history_json = json.dumps(history)
        result = self.send(history_json)
        result = json.loads(result)
        history = result["returnData"]
        return history

    def ping(self):
        ping ={
            "command": "ping"
        }
        ping_json = json.dumps(ping)
        result = self.send(ping_json)
        result = json.loads(result)
        return result["status"]

    def get_AllSymbols(self):
        allsymbols ={
            "command": "getAllSymbols"
        }
        allsymbols_json = json.dumps(allsymbols)
        result = self.send(allsymbols_json)
        result = json.loads(result)
        return result


    def get_time(self):
        time = datetime.today().strftime('%m/%d/%Y %H:%M:%S%f')
        time = datetime.strptime(time, '%m/%d/%Y %H:%M:%S%f')
        return time

    def to_milliseconds(self, days=0, hours=0, minutes=0):
        milliseconds = (days*24*60*60*1000)+(hours*60*60*1000)+(minutes*60*1000)
        return milliseconds

    def time_conversion(self, date):
        start = "01/01/1970 00:00:00"
        start = datetime.strptime(start, '%m/%d/%Y %H:%M:%S')
        date = datetime.strptime(date, '%m/%d/%Y %H:%M:%S')
        final_date = date-start
        temp = str(final_date)
        temp1, temp2 = temp.split(", ")
        hours, minutes, seconds = temp2.split(":")
        days = final_date.days
        days = int(days)
        hours = int(hours)
        hours+=2
        minutes = int(minutes)
        seconds = int(seconds)
        time = (days*24*60*60*1000)+(hours*60*60*1000)+(minutes*60*1000)+(seconds*1000)
        return time


    def is_on(self):
        temp1 = self.exec_start
        temp2 = self.get_time()
        temp = temp2 - temp1
        temp = temp.total_seconds()
        temp = float(temp)
        if temp>=8.0:
            self.connect()
        self.exec_start = self.get_time()

    def connect(self):
        try:
            self.ws=websocket.create_connection("wss://ws.xtb.com/demo")
            return True
        except:
            return False

    def disconnect(self):
        try:
            self.ws.close()
            return True
        except:
            return False

    def send(self, msg):
        self.is_on()
        self.ws.send(msg)
        result = self.ws.recv()
        return result+"\n"
    
    
def make_trades_XTB(df, pred_coln_name):
    API = XTB(XXX, XXX)

    for index, row in stqdm(df.iterrows()):
        symbol = str(row['symbol']) + '.US_4'
        current_price = API.get_tick_prices(symbol)['ask']
        lmt_buy = row['scraped_price']
        
        pred = row[pred_coln_name]
        quantity = row['quantity']
        if row['order_type'] == 'BUY' and current_price < lmt_buy:
            status, order_code = API.make_Trade(symbol = symbol, price = lmt_buy, cmd = 0, tk = pred, volume = quantity, transaction_type = 0, comment="test")
        elif current_price > lmt_buy:
            status, order_code = API.make_Trade(symbol = symbol, price = lmt_buy, cmd = 5, tk = pred, volume = quantity, transaction_type = 1, comment="test")

