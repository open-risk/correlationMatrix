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
# either express or implied. See the License for+ the specific language governing permissions and
# limitations under the License.


""" Derive a conditional correlation matrix given a stress scenario


"""
import correlationMatrix as cm

from correlationMatrix import source_path

dataset_path = source_path + "datasets/"

# Initialize a correlation matrix from the available examples
myMatrix = cm.CorrelationMatrix()

# Select method to stress

# TODO Perform PCA Analysis


# TODO Introduce A scenario shift of PCA vectors


# Specify the stress factor for all periods (in this example five)
Scenario = []

# TODO stress API
# myMatrix.stress(Scenario, Method)
