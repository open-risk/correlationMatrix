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


import numpy as np
import pandas as pd
import correlationMatrix as cm

"""
This module provides functionality for the creation of various correlation matrices 
or correlated timeseries data (in pandas dataframe format) 

Useful for controlled experiments where the generative process in fully known

"""


def generate_random_matrix(n=10):
    """
        Produce a random matrix for testing purposes
        The simple method currently implemented uses an independent factor model with random loadings
        Y = b F + e
    """

    b = np.random.uniform(low=-1.0, high=1.0, size=n)
    C = np.outer(b, b)
    for i in range(n):
        C[i, i] = 1
    myMatrix = cm.CorrelationMatrix(values=C)
    return myMatrix


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


def capm_model(n, b, sample):
    """
    Generate samples from a stylized CAPM model

    Suitable for testing estimation algorithms

    The data format is a table with columns per entity ID and a final market factor

    :param int n: The number of distinct entities to simulate
    :param list: A list of loading to the market factor
    :param int sample: The number of samples to simulate
    :rtype: pandas dataframe

    .. note:: This emulates typical downloads of data from market data API's

    """
    if len(b) != n:
        print("Inconsistent input values, length of loadings matrix differs from simulated entities")
        exit()

    errors = np.random.normal(loc=0.0, scale=1.0, size=(n, sample))
    market_factor = np.random.normal(loc=0.0, scale=1.0, size=(1, sample))
    returns = np.outer(b, market_factor) + errors
    print(returns.shape)
    print(market_factor.shape)
    print(returns.shape)
    data = np.append(returns, market_factor, 0)
    print(data.shape)
    columns = ['S' + str(i) for i in range(0, n)]
    columns.append('Market Factor')
    print(columns)
    print(data[:, :5])
    df = pd.DataFrame(data.transpose(), columns=columns)
    return df


def apt_model(n, b, m, rho, sample):
    """
    Generate samples from a stylized APT model

    Suitable for testing estimation algorithms

    The data format is a table with columns per entity ID and market factors

    :param int n: The number of distinct entities to simulate
    :param list b: A list of loading to the market factor
    :param int m: The number of market factors
    :param rho: The correlation matrix to use for the simulation of the market factors
    :param int sample: The number of samples to simulate
    :rtype: pandas dataframe

    .. note:: This emulates typical downloads of data from market data API's

    """
    if len(b) != m:
        print("Inconsistent input values, length of loadings matrix differs from simulated market factors")
        exit()

    if rho.dimension != m:
        print("Inconsistent input values, dimension of factor correlation matrix differs from simulated factors")
        exit()

    mean = np.zeros(rho.dimension)
    market_factors = np.random.multivariate_normal(mean, rho.matrix, sample).transpose()

    errors = np.random.normal(loc=0.0, scale=1.0, size=(n, sample))

    print(market_factors.shape)
    print(errors.shape)

    returns = np.dot(b, market_factors) + errors
    print(returns.shape)

    data = np.append(returns, market_factors, 0)
    print(data.shape)
    columns = ['S' + str(i) for i in range(0, n)] + ['F' + str(i) for i in range(0, m)]
    print(columns)
    print(data[:, :5])
    df = pd.DataFrame(data.transpose(), columns=columns)
    return df


def sector_model(n, b, rho, sample):
    """
    Generate samples from a stylized Sector model in the spirit of Credit Metrics

    Suitable for testing estimation algorithms

    The data format is a table with columns per entity ID and market factors

    :param int n: The number of distinct entities to simulate per sector
    :param list b: A list of loadings of the n entities to their market factor
    :param int m: The number of market factors
    :param rho: The correlation matrix to use for the simulation of the market factors
    :param int sample: The number of samples to simulate
    :rtype: pandas dataframe


    """
    # The loadings array
    b = np.array(b)

    # Sector returns on basis of input correlation matrix
    mean = np.zeros(rho.dimension)
    market_factors = np.random.multivariate_normal(mean, rho.matrix, sample).transpose()
    m = rho.dimension

    # Entity return residuals (m * n)
    errors = np.random.normal(loc=0.0, scale=1.0, size=(n * m, sample))

    # Initialize returns array (n*m)
    returns = np.zeros((n * m, sample))
    for sector in range(m):
        for entity in range(n):
            for s in range(sample):
                return_id = entity + n * sector
                returns[return_id][s] = b[entity] * market_factors[sector][s] + errors[return_id][s]

    # put entity and sector returns together
    data = np.append(returns, market_factors, 0)

    columns = ['S' + str(i) for i in range(0, n * m)] + ['F' + str(i) for i in range(0, m)]
    df = pd.DataFrame(data.transpose(), columns=columns)
    print(df.head(5))
    return df
