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
Example workflow using correlationMatrix to estimate a correlation matrix
from timeseries data in json format


"""

import pandas as pd
import numpy as np

import correlationMatrix as cm
from correlationMatrix import source_path
from correlationMatrix.utils.preprocessing import construct_returns, \
    normalize_log_returns, json_file_to_frame


input_dataset_path = source_path + "datasets/"
output_dataset_path = source_path + "datasets/"

Step = 4

if Step == 1:
    print("> Load the data from a json file")
    input_filename = output_dataset_path + 'output.json'
    output_filename = output_dataset_path + 'json_example.csv'
    json_file_to_frame(input_filename, output_filename)
elif Step == 2:
    print("> Calculate returns and save to disk")
    in_filename = output_dataset_path + 'json_example.csv'
    out_filename = output_dataset_path + 'json_example_log_returns.csv'
    construct_returns(in_filename, out_filename)
elif Step == 3:
    print("> Calculate normalized returns and save to disk")
    in_filename = output_dataset_path + 'json_example_log_returns.csv'
    out_filename = output_dataset_path + 'json_example_scaled_returns.csv'
    mean, std = normalize_log_returns(in_filename, out_filename)
    print('Normalized Means: ', mean)
    print('Normalized Stddev: ', std)
elif Step == 4:
    print("> Calculate empirical correlation matrix")
    in_filename = output_dataset_path + 'json_example_scaled_returns.csv'
    myMatrix = cm.EmpiricalCorrelationMatrix()
    data = pd.read_csv(in_filename)
    myMatrix.fit(data)
    myMatrix.print()
    L = np.linalg.cholesky(myMatrix.matrix)
    print(L)
