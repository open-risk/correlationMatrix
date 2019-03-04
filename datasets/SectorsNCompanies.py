"""
Select Equities from Yahoo! Finance
- 5 out of 9 industry sectors
- 10 companies from each sector (sample)

This symbol list is valid (contains data) as of February 2019
"""

yahoo_names = [
    (0, 'Financial Index', "AXP", "American Express"),
    (1, 'Financial Index', "JPM", "JP Morgan Chase"),
    (2, 'Financial Index', "WFC", "Wells Fargo"),
    (3, 'Financial Index', "BAC", "Bank of America Corp"),
    (4, 'Financial Index', "C", "Citigroup"),
    (5, 'Financial Index', "USB", "US Bancorp"),
    (6, 'Financial Index', "GS", "Goldman Sachs Group"),
    (7, 'Financial Index', "MS", "Morgan Stanley"),
    (8, 'Financial Index', "ALL", "Allstate Corporation"),
    (9, 'Financial Index', "AIG", "American International Group"),
    (10, 'Health Care', "UNH", "Unitedhealth Group Inc"),
    (11, 'Health Care', "CVS", "CVS Health"),
    (12, 'Health Care', "CAH", "Cardinal Health"),
    (13, 'Health Care', "PFE", "Pfizer Inc."),
    (14, 'Health Care', "NVS", "Novartis AG"),
    (15, 'Health Care', "CI", "Cigna Corporation"),
    (16, 'Health Care', "HUM", "Humana Inc"),
    (17, 'Health Care', "LH", "Lab Corp of America Hldgs"),
    (18, 'Health Care', "DGX", "Quest Diagnostics"),
    (19, 'Health Care', "UHS", "Universal Health Services B"),
    (20, 'Technology', "QCOM", "QUALCOMM Inc"),
    (21, 'Technology', "AAPL", "Apple Inc."),
    (22, 'Technology', "HPQ", "Hewlett Packard"),
    (23, 'Technology', "AMZN", "Amazon Inc."),
    (24, 'Technology', "MSFT", "Microsoft Corp"),
    (25, 'Technology', "GOOGL", "Alphabet Inc A"),
    (26, 'Technology', "INTC", "Intel Corp"),
    (27, 'Technology', "CSCO", "Cisco Systems Inc"),
    (28, 'Technology', "IBM", "Intl Business Machines Corp"),
    (29, 'Technology', "ORCL", "Oracle Corp"),
    (30, 'Oil & Gas', "APC", "Anadarko Petroleum Corp"),
    (31, 'Oil & Gas', "XOM", "Exxon Mobil Corp"),
    (32, 'Oil & Gas', "CVX", "Chevron Corp"),
    (33, 'Oil & Gas', "SLB", "Schlumberger Ltd"),
    (34, 'Oil & Gas', "OXY", "Occidental Petroleum"),
    (35, 'Oil & Gas', "COP", "ConocoPhillips"),
    (36, 'Oil & Gas', "EOG", "EOG Resources"),
    (37, 'Oil & Gas', "HAL", "Halliburton Co"),
    (38, 'Oil & Gas', "VLO", "Valero Energy COrporation"),
    (39, 'Oil & Gas', "PXD", "Pinoeer Natural Resources"),
    (40, 'Consumer Goods', "F", "Ford Motor Co"),
    (41, 'Consumer Goods', "PG", "Procter & Gamble"),
    (42, 'Consumer Goods', "KO", "Coca-Cola Co"),
    (43, 'Consumer Goods', "PEP", "PepsiCo Inc"),
    (44, 'Consumer Goods', "MO", "Altria Group Inc"),
    (45, 'Consumer Goods', "MDLZ", "Mondelez International Inc"),
    (46, 'Consumer Goods', "NKE", "NIKE Inc B"),
    (47, 'Consumer Goods', "CL", "Colgate-Palmolive Co"),
    (48, 'Consumer Goods', "GIS", "General Mills"),
    (49, 'Consumer Goods', "ADM", "Archer Daniels Midland")
]

