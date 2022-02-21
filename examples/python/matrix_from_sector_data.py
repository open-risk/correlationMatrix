# encoding: utf-8

# (c) 2019-2022 Open Risk, all rights reserved
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
Example workflow using correlationMatrix to estimate a correlation matrix
from equity data belonging to different sectors (Credit Metrics model)

Data are assumed already available locally on disk
Use utils/fetch_equity_data.py to fetch Yahoo Finance equity data

"""

import pandas as pd

import correlationMatrix as cm
from correlationMatrix import source_path
from correlationMatrix.utils.preprocessing import csv_files_to_frame, construct_log_returns, \
    normalize_log_returns
from datasets.SectorsNCompanies import yahoo_names as entity_list

input_dataset_path = source_path + "datasets/yahoo_equity_data/"
output_dataset_path = source_path + "datasets/"

Step = 2

if Step == 1:
    # Concatenate input data into a single dataframe and save to disk
    print("> Concatenate input data into a single dataframe and save to disk")
    filename = output_dataset_path + 'yahoo_merged_data.csv'
    csv_files_to_frame(entity_list, input_dataset_path, filename)
elif Step == 2:
    print("> Calculate log returns and save to disk")
    in_filename = output_dataset_path + 'yahoo_merged_data.csv'
    out_filename = output_dataset_path + 'yahoo_log_returns.csv'
    # Drop any columns we don't want to compute differences on
    drop_columns = ['Date']
    construct_log_returns(in_filename, out_filename, drop_columns)
elif Step == 3:
    print("> Calculate normalized log returns and save to disk")
    in_filename = output_dataset_path + 'yahoo_log_returns.csv'
    out_filename = output_dataset_path + 'yahoo_scaled_returns.csv'
    normalize_log_returns(in_filename, out_filename)
elif Step == 4:
    print("> Calculate sectoral factor model")
    in_filename = output_dataset_path + 'yahoo_scaled_returns.csv'
    myMatrix = cm.FactorCorrelationMatrix()
    data = pd.read_csv(in_filename)
    myMatrix.fit(data, method='CreditMetrics')


