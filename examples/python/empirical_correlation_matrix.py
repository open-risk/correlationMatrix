# encoding: utf-8

# (c) 2019-2024 Open Risk, all rights reserved
#
# correlationMatrix is licensed under the Apache 2.0 license a copy of which is included
# in the source distribution of correlationMatrix. This is notwithstanding any licenses of
# third-party software included in this distribution. You may not use this file except in
# compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specif ic language governing permissions and
# limitations under the License.


"""
Example workflow using correlationMatrix to estimate an empirical correlation matrix from
timeseries data. The datasets are produced in examples/generate_synthetic_data.py

"""
import pandas as pd

import correlationMatrix as cm
from correlationMatrix import source_path

dataset_path = source_path + "datasets/"

print("> Step 1: Load the data set into a pandas frame")
data = pd.read_csv(dataset_path + 'synthetic_data1.csv')
# Estimate the empirical correlation matrix using the Pearson measure
print("> Step 3a: Estimate the empirical correlation matrix using the Pearson measure")
myMatrix = cm.EmpiricalCorrelationMatrix()
# print(myMatrix.validated)
# print(type(myMatrix))
# myMatrix.print()
# myMatrix.pearsonr(data)
myMatrix.fit(data, method='pearson')
myMatrix.print()
print("> Step 3b: Estimate the empirical correlation matrix using the Kendall measure")
myMatrix.fit(data, method='kendall')
myMatrix.print()
print("> Step 3c: Estimate the empirical correlation matrix using the Spearman measure")
myMatrix.fit(data, method='spearman')
myMatrix.print()
