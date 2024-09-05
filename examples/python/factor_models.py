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
# either express or implied. See the License for the specific language governing permissions and
# limitations under the License.


"""
Example workflows using correlationMatrix to estimate factor models from
timeseries data. The datasets are produced in examples/generate_synthetic_data.py

"""
import pandas as pd

import correlationMatrix as cm
from correlationMatrix import source_path

dataset_path = source_path + "datasets/"

# Select which example you want to explore

# Example 1: Uniform correlation multivariate model
# Example 2: CAPM style Model
# Example 3: APT style Model with uncorrelated market factors
# Example 4: Credit Metrics style factor model

example = 4

# Step 1
# Load the data set into a pandas frame
# Make sure state is read as a string and not as integer
# Second synthetic data example:
# n entities with ~10 observations each, [0,1] state, 50%/50% correlation matrix
print("> Step 1: Load the data set into a pandas frame")
if example == 1:
    data = pd.read_csv(dataset_path + 'synthetic_data1.csv')
    print("> Step 2: Estimate Uniform single factor model")
    myMatrix = cm.FactorCorrelationMatrix()
    myMatrix.fit(data, method='UniformSingleFactor')
    # myMatrix.print()
elif example == 2:
    data = pd.read_csv(dataset_path + 'synthetic_data2.csv')
    print("> Step 2: Estimate CAPM style model")
    myMatrix = cm.FactorCorrelationMatrix()
    myMatrix.fit(data, method='CAPMModel')
elif example == 3:
    data = pd.read_csv(dataset_path + 'synthetic_data3.csv')
    print("> Step 2: Estimate APT style model")
    myMatrix = cm.FactorCorrelationMatrix()
    myMatrix.fit(data, method='APTModel')
elif example == 4:
    data = pd.read_csv(dataset_path + 'synthetic_data4.csv')
    print("> Step 2: Estimate Credit Metrics style model")
    myMatrix = cm.FactorCorrelationMatrix()
    myMatrix.fit(data, method='CreditMetrics')
