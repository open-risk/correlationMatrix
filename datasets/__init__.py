# encoding: utf-8

# (c) 2019-2023 Open Risk, all rights reserved
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
Predefined correlation matrices and other useful data
"""

# A Minimal matrix (uniform)

Minimal = [
    [1.0, 0.2, 0.2],
    [0.2, 1.0, 0.2],
    [0.2, 0.2, 1.0]
]

# Vandermonde correlation is based (loosely) on the Vandermonde matrix
# https://en.wikipedia.org/wiki/Vandermonde_matrix
# the correlation drops exponentially with the distance

Vandermonde = [
    [1.0, 0.5, 0.25, 0.0625],
    [0.5, 1.0, 0.5, 0.25],
    [0.25, 0.5, 1.0, 0.5],
    [0.0625, 0.25, 0.5, 1.0]
]