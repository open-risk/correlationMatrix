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


""" Validate a set of calculated thresholds

"""

import correlationMatrix as cm
from datasets import Generic
from correlationMatrix.thresholds.model import ThresholdSet
from correlationMatrix.thresholds.settings import AR_Model
from correlationMatrix import source_path
dataset_path = source_path + "datasets/"

# Initialize a single period correlation matrix
# Example 1: Generic -> Typical Credit Rating correlation Matrix
# Example 2: Minimal -> Three state correlation matrix

M = cm.CorrelationMatrix(values=Generic)
# Lets take a look at the values
M.print()
M.validate()

# The size of the rating scale
Ratings = M.dimension

# The Default (absorbing state)
Default = Ratings - 1

# Lets extend the matrix into multi periods
Periods = 5
T = cm.CorrelationMatrixSet(values=M, periods=Periods, method='Power', temporal_type='Cumulative')

# Initialize a threshold set
As = ThresholdSet(TMSet=T)

print("> Fit Multiperiod Thresholds")
for ri in range(0, Ratings):
    print("RI: ", ri)
    As.fit(AR_Model, ri)

print("> Validate Multiperiod Thresholds against Input correlation Matrix Set")
Q = As.validate(AR_Model)

print("> Save Multiperiod Thresholds in JSON Format")
As.to_json('generic_thresholds.json')
