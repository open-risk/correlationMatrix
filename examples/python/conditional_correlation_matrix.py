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
# either express or implied. See the License for+ the specific language governing permissions and
# limitations under the License.


""" Derive a conditional correlation matrix given a stress scenario


"""
import numpy as np
import correlationMatrix as cm
from datasets import Generic
from correlationMatrix import source_path
dataset_path = source_path + "datasets/"


# Initialize a threshold set from file
As = ThresholdSet(json_file=dataset_path + 'generic_thresholds.json')

# Inspect values (we assume these inputs have already been validated after generation!)
# As.print(accuracy=4)

# Specify the initial rating of interest
ri = 3

# As.plot(ri)

# Initialize a conditional migration matrix with the given thresholds

Q = cm.ConditionalcorrelationMatrix(thresholds=As)

# # Q.print()
#
# print(dir(Q))
#
# Specify the stress factor for all periods (in this example five)
Scenario = np.zeros((Q.periods), dtype=float)

Scenario[0] = 2.0
Scenario[1] = 2.0
Scenario[2] = - 2.0
Scenario[3] = - 2.0
Scenario[4] = 0.0

# Specify sensitivity to stress
rho = 0.5

# Calculate conditional correlation rates for an initial state (5)
Q.fit(AR_Model, Scenario, rho, ri)
# Print the conditional correlation rates for that rating
Q.print_matrix(format_type='Standard', accuracy=4, state=ri)
# Graph the modelled survival densities versus migration thresholds
Q.plot_densities(state=ri)
# Q.plot_densities(1, ri)
