# encoding: utf-8

# (c) 2019 Open Risk, all rights reserved
#
# correlationMatrix is licensed under the Apache 2.0 license a copy of which is included
# in the source distribution of correlationMatrix. This is notwithstanding any licenses of
# third-party software included in this distribution. You may not use this file except in
# compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.

"""
    Script for downloading Yahoo Finance data
    The symbols are imported from the datasets/SectorsNCompanies dictionary

    Timeseries data are stored in individual files within a directory

"""


import pandas_datareader as pdr
from datetime import datetime
import time

from datasets.SectorsNCompanies import yahoo_names
from correlationMatrix import source_path
dataset_path = source_path + "datasets/yahoo_equity_data"


for entity in yahoo_names[13:]:
    symbol = entity[2]
    dataset = pdr.get_data_yahoo(symbols=symbol, start=datetime(2000, 1, 1), end=datetime(2019, 1, 1))
    dataset.to_csv(dataset_path + "/" + symbol + ".csv")
    time.sleep(2)
    print(symbol)

# For individual stock testing
# symbol = "NVS"
# dataset = pdr.get_data_yahoo(symbols=symbol, start=datetime(2000, 1, 1), end=datetime(2019, 1, 1))
# dataset.to_csv(dataset_path + "/" + symbol + ".csv")

