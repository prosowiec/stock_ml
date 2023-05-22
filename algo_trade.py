from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order_condition import Create, OrderCondition
from ibapi.order import *
from stqdm import stqdm
import threading
import time

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


def make_trades(df, pred_coln_name):
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