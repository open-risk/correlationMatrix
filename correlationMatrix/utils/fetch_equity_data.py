# from yahoo_finance import Share
# yahoo = Share('YHOO')
# stock_data =  yahoo.get_historical('2014-04-25', '2014-04-29')
# print(stock_data)

# import urllib.request
# import time
#
# stockstoPull = 'AMD', 'BAC', 'MSFT', 'TXN', 'GOOG'
#
# def pullData(stock):
#     fileLine = stock + '.txt'
#     urltovisit = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=1y/csv'
#     with urllib.request.urlopen(urltovisit) as f:
#         sourceCode = f.read().decode('utf-8')
#     splitSource = sourceCode.split('\n')
#
#     for eachLine in splitSource:
#         splitLine = eachLine.split(',') # <---(here ',' instead of '.')
#         if len(splitLine) == 6: # <----( here, 6 instead of 5 )
#             if 'values' not in eachLine:
#                 saveFile = open(fileLine,'a')
#                 linetoWrite = eachLine+'\n'
#                 saveFile.write(linetoWrite)
#
#     print('Pulled', stock)
#     print('...')
#     time.sleep(.5)
#
# if __name__=="__main__":
#     for eachStock in stockstoPull:
#         pullData(eachStock)

"""
Select Equities from Yahoo! Finance

5 sectors, 10 companies for each
"""

vnames = []

""" Financial Index """

symbols = ["AXP",  # American Express
           "JPM",  # JP Morgan Chase
           "WFC",  # Wells Fargo
           "BAC",  # Bank of America Corp
           "C",  # Citigroup
           "USB",  # US Bancorp
           "GS",  # Goldman Sachs Group
           "MS",  # Morgan Stanley
           "ALL",  # Allstate Corporation
         "AIG"]  # American International Group

vnames.append(symbols)


""" Health Care """

symbols = ["UNH",  # Unitedhealth Group Inc
             "CVS",  # CVS Health
             "CAH",  # Cardinal Health
             "ESRX",  # Express Scripts Holding Co.
             "AET",  # Aetna Inc
             "CI",  # Cigna Corporation
             "HUM",  # Humana Inc
             "LH",  # Lab Corp of America Hldgs
             "DGX",  # Quest Diagnostics
             "UHS"]  # Universal Health Services B

vnames.append(symbols)

""" Technology """

symbols = ["AAPL",  # Apple Inc.
             "HPQ",  # Hewlett Packard
             "AMZN",  # Amazon Inc.
             "MSFT",  # Microsoft Corp
             "GOOGL",  # Alphabet Inc A
             "INTC",  # Intel Corp
             "CSCO",  # Cisco Systems Inc
             "IBM",  # Intl Business Machines Corp
             "ORCL",  # Oracle Corp
             "QCOM"]  # QUALCOMM Inc

vnames.append(symbols)

""" Oil & Gas """

symbols = ["XOM",  # Exxon Mobil Corp
             "CVX",  # Chevron Corp
             "SLB",  # Schlumberger Ltd
             "OXY",  # Occidental Petroleum
             "COP",  # ConocoPhillips
             "EOG",  # EOG Resources
             "HAL",  # Halliburton Co
             "VLO",  # Valero Energy COrporation
             "PXD",  # Pinoeer Natural Resources
             "APC"]  # Anadarko Petroleum Corp

vnames.append(symbols)

"""  Consumer Goods """

symbols = ["PG",  # Procter & Gamble
             "KO",  # Coca-Cola Co
             "PEP",  # PepsiCo Inc
             "MO",  # Altria Group Inc
             "MDLZ",  # Mondelez International Inc
             "NKE",  # NIKE Inc B
             "CL",  # Colgate-Palmolive Co
             "GIS",  # General Mills
             "ADM",  # Archer Daniels Midland
             "F"]  # Ford Motor Co

vnames.append(symbols)

import pandas_datareader as pdr
from datetime import datetime

print(vnames)

# appl = pdr.get_data_yahoo(symbols='AAPL', start=datetime(2000, 1, 1), end=datetime(2019, 1, 1))
# print(appl['Adj Close'])
