# Data provided for free by IEX. View IEX’s Terms of Use.

import sys,requests
import pandas as pd
import time
from datetime import date,timedelta
#import socket

pd.set_option('max_columns',5)

class stock_info(object):
	def __init__(self,ticker):
		self.ticker=ticker
		self.range=None

		#check that ticker exists
		try:
			self.get_data('range')
		except:
			raise ValueError('Invalid Ticker')

	def findNextDay(self):
		''' finds next yyyymmdd based on current Date '''

		t=time.strptime(str(self.date),'%Y%m%d')
		newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(1)
		next =  newdate.strftime('%Y%m%d')
		return next

	def get_data(self,data_type):
		''' collects minute data from current day '''

		if data_type=='date':
			s='https://api.iextrading.com/1.0/stock/%s/chart/date/%s' % (self.ticker,self.date)
		elif data_type=='range':
			s='https://api.iextrading.com/1.0/stock/%s/chart/%s' % (self.ticker,self.range)
		elif data_type=='multiple':
			s='https://api.iextrading.com/1.0/stock/market/batch?symbols=%s,%s\
			&types=quote,news,chart&range=1m&last=5' % (self.ticker,self.ticker2)
		if data_type=='today':
			s='https://api.iextrading.com/1.0/stock/%s/quote' % self.ticker
		print(s)
		data=requests.get(s).json()
		return data

	def create_minute_dataset(self,start,end=0):
		'''creates pandas Data Frame of minute data from start date to end date'''

		self.date=start
		data=self.get_data('date')
		if end!=0:
			while self.date<=end:
				self.date=self.findNextDay()
				nextDay=self.get_data('date')
				data=data+nextDay

		df=pd.DataFrame(data)
		df=df.set_index('minute')
		return df

	def create_ranged_dataset(self,range):
		'''choose range of 5y,2y,ytd,6m,3m,1m,1d'''

		if range not in ['5y','2y','ytd','6m','3m','1m','1d']:
			raise ValueError('Invalid Range')
		self.range=range
		data=self.get_data('range')
		df=pd.DataFrame(data)
		if range=='1d':
			df=df.set_index('minute')
		else:
			df=df.set_index('date')
		return df

	def return_today_data(self):
		'''Grabs stock data for today'''
		data=self.get_data('today')
		df=pd.DataFrame([data])
		df=df.set_index('symbol')
		return df

if __name__=='__main__':
	print("Type in Stock Ticker. For more information, refer to readme")
	data = sys.stdin.readline().split(',')
	#example: data = "FB,20190101,20190105".split(',')
	stock=stock_info(data[0].strip())
	if len(data)==1:
		#today's data
		stockinfo=stock.return_today_data()
	elif len(data)==3:
		#minute data for one day
		stockinfo = stock.create_minute_dataset(data[1].strip(),data[2].strip())
	elif len(data[1].strip())==8 and len(data)==2:
		#minute data for a range of days
		stockinfo = stock.create_minute_dataset(data[1].strip())
	elif len(data)==2:
		#daily data for specified range
		stockinfo =stock.create_ranged_dataset(data[1].strip())
	else:
		raise ValueError("Expect 1,2, or 3 arguments")
	print(stockinfo)
