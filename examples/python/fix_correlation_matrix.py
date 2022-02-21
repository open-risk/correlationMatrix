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


""" Example of using correlationMatrix to detect and solve various pathologies that might be affecting correlation
matrix data

TODO this will need major expansion to implement nearest correlation matrix algorithms

"""

import correlationMatrix as cm

print("> Initialize an invalid matrix")
B = cm.CorrelationMatrix(values=[[-0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
B.print()

B.fix_negative_values()
B.print()
