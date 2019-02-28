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


"""
Example workflows using correlationMatrix to generate synthetic data

1.
2.
3.
4.


"""

import pandas as pd
import correlationMatrix as cm
from datasets import Generic
from correlationMatrix import source_path
from correlationMatrix.utils import dataset_generators

dataset_path = source_path + "datasets/"

# select the data set to produce
dataset = 1

#
# Duration type datasets in Compact Format
#
if dataset == 1:
    # This dataset creates the simplest possibe (uniform single factor) correlation matrix
    # Correlation Matrix definition
    # n: number of entities to generate
    myMatrix = cm.CorrelationMatrix(type='UniformSingleFactor', rho=0.3, n=10)
    # myMatrix.print()
    # Generate multivariate normal data with that correlation matrix (a pandas frame)
    # s: number of samples per entity
    data = dataset_generators.multivariate_normal(myMatrix, sample=100)
    data.to_csv(dataset_path + 'synthetic_data1.csv', index=False)

elif dataset == 2:
    # Second example: Multiple Entities observed over continuous short time interval
    myState = cm.StateSpace([('0', "Basic"), ('1', "Default")])
    data = dataset_generators.exponential_correlations(myState, n=1000, sample=10, rate=0.1)
    sorted_data = data.sort_values(['ID', 'Time'], ascending=[True, True])
    sorted_data.to_csv(dataset_path + 'synthetic_data2.csv', index=False)

elif dataset == 3:
    # Third example: Multiple Entities with Medium sized State space observed over continuous long time interval
    myState = cm.StateSpace([('0', "A"), ('1', "B"), ('2', "C"), ('3', "D"), ('4', "E"), ('5', "F"), ('6', "G")])
    data = dataset_generators.exponential_correlations(myState, n=100, sample=20, rate=0.1)
    sorted_data = data.sort_values(['ID', 'Time'], ascending=[True, True])
    sorted_data.to_csv(dataset_path + 'synthetic_data3.csv', index=False)

#
# Cohort type datasets
#
elif dataset == 4:
    # Fourth example: S&P Credit Rating Migration Matrix
    # S&P Ratings State Space
    definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                   ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]

    matrix = Generic
    myState = cm.StateSpace(definition)
    # Fourth example: Single entity observed over discrete short time interval
    data = dataset_generators.markov_chain(myState, matrix, n=1000, timesteps=10)
    sorted_data = data.sort_values(['ID', 'Timestep'], ascending=[True, True])
    sorted_data.to_csv(dataset_path + 'synthetic_data4.csv', index=False)

elif dataset == 5:
    # Fifth example: An IFRS 9 style Migration Matrix for migrations between Stages
    definition = [('0', "Stage 1"), ('1', "Stage 2"), ('2', "Stage 3")]

    matrix = [[0.8, 0.15, 0.05],
              [0.1, 0.7, 0.2],
              [0.0, 0.0, 1.0]]

    myState = cm.StateSpace(definition)
    data = dataset_generators.markov_chain(myState, correlationmatrix=matrix, n=10000, timesteps=5)
    sorted_data = data.sort_values(['ID', 'Timestep'], ascending=[True, True])
    sorted_data.to_csv(dataset_path + 'synthetic_data5.csv', index=False)

elif dataset == 6:
    # Simplest absorbing state case for validation purposes
    myState = cm.StateSpace()
    myState.generic(2)
    myState.describe()
    matrix = [[0.5, 0.5],
              [0.0, 1.0]]
    data = dataset_generators.markov_chain(myState, correlationmatrix=matrix, n=1000, timesteps=20)
    sorted_data = data.sort_values(['ID', 'Timestep'], ascending=[True, True])
    data.to_csv(dataset_path + 'synthetic_data6.csv', index=False)

#
# Duration type datasets in Long Format
#

elif dataset == 7:
    # Seventh example: Credit Rating Migrations in Long Format
    definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                   ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]
    myState = cm.StateSpace(definition)
    matrix = Generic
    # Timesteps are the interval periods over which to repeatedly simulate the single period migration matrix
    data = dataset_generators.long_format(myState, correlationmatrix=matrix, n=1000, timesteps=10)
    sorted_data = data.sort_values(['Time', 'ID', 'From'], ascending=[True, True, True])
    sorted_data.to_csv(dataset_path + 'synthetic_data7.csv', index=False)

elif dataset == 8:
    # Simplest absorbing state case for validation purposes (Duration estimator)
    # Single timestep
    definition = [('0', "G"), ('1', "B")]
    myState = cm.StateSpace()
    myState.generic(2)
    myState.describe()
    matrix = [[0.5, 0.5],
              [0.0, 1.0]]
    data = dataset_generators.long_format(myState, correlationmatrix=matrix, n=10000, timesteps=2)
    sorted_data = data.sort_values(['Time', 'ID', 'From'], ascending=[True, True, True])
    data.to_csv(dataset_path + 'synthetic_data8.csv', index=False)

elif dataset == 9:
    # Ninth example: Credit Rating Migrations in Long Format with Monthly Observations
    # Inputs are:
    # - the State Space (Entries in the From and To columns)
    # - the Annual correlation Matrix (probabilities)
    # - the number of entities to simulate
    # - the start date of the Observation Window in DD-MM-YY format

    definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                   ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]
    myState = cm.StateSpace(definition)
    matrix = Generic
    origin = '2006-01-01'
    # Timesteps are the interval periods over which to repeatedly simulate the single period migration matrix
    # The correlation times are decimals in the range [0, N-1]
    data = dataset_generators.long_format(myState, correlationmatrix=matrix, n=1000, timesteps=10)
    # Convert to daily units
    data['Time'] *= 365
    sorted_data = data.sort_values(['Time', 'ID', 'From'], ascending=[True, True, True])
    # Convert decimals to datetime format
    sorted_data['Time'] = pd.to_datetime(sorted_data['Time'], unit='D', origin=origin, errors='coerce')
    # Remove intra-day information
    sorted_data['Time'] = sorted_data['Time'].apply(lambda x: x.date())

    sorted_data.to_csv(dataset_path + 'synthetic_data9.csv', index=False)

print("> Synthetic Dataset:", dataset, " has been created and stored in the filesystem.")
