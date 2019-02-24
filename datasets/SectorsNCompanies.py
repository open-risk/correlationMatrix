"""
Select Equities from Yahoo! Finance

5 sectors, 10 companies for each
"""

vnames = []

""" Financial Index """

vnames[1] = ["AXP",  # American Express
             "JPM",  # JP Morgan Chase
             "WFC",  # Wells Fargo
             "BAC",  # Bank of America Corp
             "C",  # Citigroup
             "USB",  # US Bancorp
             "GS",  # Goldman Sachs Group
             "MS",  # Morgan Stanley
             "ALL",  # Allstate Corporation
             "AIG"]  # American International Group

""" Health Care """

vnames[2] = ["UNH",  # Unitedhealth Group Inc
             "CVS",  # CVS Health 
             "CAH",  # Cardinal Health 
             "ESRX",  # Express Scripts Holding Co.
             "AET",  # Aetna Inc
             "CI",  # Cigna Corporation
             "HUM",  # Humana Inc
             "LH",  # Lab Corp of America Hldgs
             "DGX",  # Quest Diagnostics
             "UHS"]  # Universal Health Services B

""" Technology """

vnames[3] = ["AAPL",  # Apple Inc.
             "HPQ",  # Hewlett Packard
             "AMZN",  # Amazon Inc.
             "MSFT",  # Microsoft Corp
             "GOOGL",  # Alphabet Inc A
             "INTC",  # Intel Corp
             "CSCO",  # Cisco Systems Inc
             "IBM",  # Intl Business Machines Corp
             "ORCL",  # Oracle Corp
             "QCOM"]  # QUALCOMM Inc

""" Oil & Gas """

vnames[4] = ["XOM",  # Exxon Mobil Corp
             "CVX",  # Chevron Corp
             "SLB",  # Schlumberger Ltd
             "OXY",  # Occidental Petroleum
             "COP",  # ConocoPhillips
             "EOG",  # EOG Resources
             "HAL",  # Halliburton Co
             "VLO",  # Valero Energy COrporation
             "PXD",  # Pinoeer Natural Resources  
             "APC"]  # Anadarko Petroleum Corp

"""  Consumer Goods """

vnames[5] = ["PG",  # Procter & Gamble
             "KO",  # Coca-Cola Co
             "PEP",  # PepsiCo Inc
             "MO",  # Altria Group Inc
             "MDLZ",  # Mondelez International Inc
             "NKE",  # NIKE Inc B
             "CL",  # Colgate-Palmolive Co
             "GIS",  # General Mills
             "ADM",  # Archer Daniels Midland
             "F"]  # Ford Motor Co
