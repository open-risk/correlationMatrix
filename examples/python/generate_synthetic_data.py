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
Example workflows using correlationMatrix to generate synthetic data

1. Uniform single factor correlations among a number of entities
2. Multiple entities correlated with a single market factor
3. Multiple entities correlated with multiple uncorrelated factors
4. Multiple entities correlated with multiple correlated factors


"""

import numpy as np

import correlationMatrix as cm
from correlationMatrix import source_path
from correlationMatrix.utils import dataset_generators

dataset_path = source_path + "datasets/"

# select the data set to produce
dataset = 4

#
# Duration type datasets in Compact Format
#
if dataset == 1:
    # This dataset creates the simplest possibe (uniform single factor) correlation matrix
    # Correlation Matrix definition
    # n: number of entities to generate
    myMatrix = cm.CorrelationMatrix(type='UniformSingleFactor', rho=0.3, n=50)
    # myMatrix.print()
    # Generate multivariate normal data with that correlation matrix (a pandas frame)
    # s: number of samples per entity
    data = dataset_generators.multivariate_normal(myMatrix, sample=1000)
    data.to_csv(dataset_path + 'synthetic_data1.csv', index=False)

elif dataset == 2:
    # This dataset creates a correlation matrix between entities and an exogenous market factor
    # "CAPM Style"
    # n: number of entities
    # b: vector of loadings
    # s: number of samples
    b = [0.2, 0.3, 0.5]
    data = dataset_generators.capm_model(n=len(b), b=b, sample=10000)
    data.to_csv(dataset_path + 'synthetic_data2.csv', index=False)

elif dataset == 3:
    # This dataset creates a correlation matrix between entities and a set of exogenous market factors
    # NOTE: The market factors are assumed uncorrelated - this is useful for testing purposes
    # "APT Style"
    # n: number of entities
    # b: vector of loadings of entities to different market factors (assumed uniform)
    # rho: correlation matrix of factors of size m
    # s: number of samples
    b = [0.2, 0.2, 0.2]
    m = 3
    rho = cm.CorrelationMatrix(type='UniformSingleFactor', rho=0.0, n=m)
    data = dataset_generators.apt_model(n=10, b=b, m=len(b), rho=rho, sample=10000)
    data.to_csv(dataset_path + 'synthetic_data3.csv', index=False)

elif dataset == 4:
    # This dataset creates a sector based correlation matrix between entities
    # emulating the well known Credit Metrics model
    # NOTE: The sector factors are assumed correlated with uniform correlation
    # this is useful for testing purposes
    # n: number of entities per sector
    # m: number of sectors
    # rho: correlation matrix of factors of size m
    # s: number of samples
    n = 10
    m = 5
    # b: vector of loadings of entities to the different sectors.
    # For simplicity each entity loads only to one sectoral factor and
    # all loadings are equal
    b = list(0.2 * np.ones(n))
    rho = cm.CorrelationMatrix(type='UniformSingleFactor', rho=0.3, n=m)
    data = dataset_generators.sector_model(n=n, b=b, rho=rho, sample=10000)
    data.to_csv(dataset_path + 'synthetic_data4.csv', index=False)
