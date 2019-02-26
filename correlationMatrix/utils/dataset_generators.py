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

# library to create synthetic correlated datasets
#

import numpy as np
import pandas as pd


def multivariate_normal(correlationmatrix, sample):
    """
    Generate samples sampling from a multivariate normal.

    Suitable for testing cohorting algorithms and duration based estimators.

    The data format is a table with columns per entity ID

    :param correlationmatrix: The correlatio matrix use for the simulation
    :param int n: The number of distinct entities to simulate
    :param int sample: The number of samples to simulate
    :rtype: pandas dataframe

    .. note:: This emulates typical downloads of data from market data API's

    """

    mean = np.zeros(correlationmatrix.dimension)
    data = np.random.multivariate_normal(mean, correlationmatrix.matrix, sample)
    columns = ['S' + str(i) for i in range(0, correlationmatrix.dimension)]
    return pd.DataFrame(data, columns=columns)
