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
# either express or implied. See the License for the specif ic language governing permissions and
# limitations under the License.


"""
Example workflows using correlationMatrix to estimate an empirical correlation matrix from
timeseries data. The datasets are produced in examples/generate_synthetic_data.py

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import correlationMatrix as cm
from correlationMatrix import source_path
from correlationMatrix.utils.converters import datetime_to_float

dataset_path = source_path + "datasets/"

# Example 1: Credit Rating Migration example
# Example 2: Simple 2x2 Matrix for testing
# Example 3: Credit Rating Migration example with timestamps in raw date format

example = 1

# Step 1
# Load the data set into a pandas frame
# Make sure state is read as a string and not as integer
# Second synthetic data example:
# n entities with ~10 observations each, [0,1] state, 50%/50% correlation matrix
print("> Step 1: Load the data set into a pandas frame")
if example == 1:
    data = pd.read_csv(dataset_path + 'synthetic_data1.csv')
elif example == 2:
    data = pd.read_csv(dataset_path + 'synthetic_data8.csv', dtype={'State': str})
elif example == 3:
    data = pd.read_csv(dataset_path + 'synthetic_data9.csv', parse_dates=True)
    # convert datetime data to floats, return also the observation window data
    bounds, data = datetime_to_float(data)
    print('Start and End dates', bounds)

# print(data.head(5))
# print(data.describe())
# print(len(data.index))


# Step
# Estimate the empirical correlation matrix using the Pearson measure
print("> Step 3a: Estimate the empirical correlation matrix using the Pearson measure")
myMatrix = cm.EmpiricalCorrelationMatrix()
# print(myMatrix.validated)
# print(type(myMatrix))
# myMatrix.print()
# myMatrix.pearsonr(data)
myMatrix.fit(data, method='pearson')
myMatrix.print()
print("> Step 3b: Estimate the empirical correlation matrix using the Kendall measure")
myMatrix.fit(data, method='kendall')
myMatrix.print()
print("> Step 3c: Estimate the empirical correlation matrix using the Spearman measure")
myMatrix.fit(data, method='spearman')
myMatrix.print()
# myMatrix.validate()
# print(myMatrix.validated)
# myEstimator = aj.AalenJohansenEstimator(states=myState)
# labels = {'Timestamp': 'Time', 'From_State': 'From', 'To_State': 'To', 'ID': 'ID'}
# etm, times = myEstimator.fit(sorted_data, labels=labels)

"""
# Step 4
# Print the cumulative computed matrix
print("> Step 4: Print the cumulative computed matrix")
print(etm[:, :, -1])

# Step 5
# Create a visualization of the correlation rates
if example == 1 or example == 3:
    # Now lets plot a collection of curves for all ratings
    print("> Plot the correlation curves")

    Periods = 10
    Ratings = 8

    m = 4
    n = 2
    f, axarr = plt.subplots(m, n)
    f.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.90, wspace=0.0, hspace=0.1)
    # plt.style.use(['ggplot'])

    for ri in range(0, Ratings):
        axj = int(ri / 2)
        axi = ri % 2
        print(ri, axj, axi)
        curves = []
        for rf in range(0, Ratings):
            cPD = etm[ri, rf, :]
            curves.append(cPD)
            # axarr[axj, axi].set_aspect(5)
            axarr[axj, axi].set_ylabel('State ' + str(ri), fontsize=12)
            axarr[axj, axi].set_xlabel("Time")
            axarr[axj, axi].plot(times[1:], curves[rf], label="RI=%d" % (rf,))
            # axarr[axj, axi].set_xticks(range(10), minor=False)
            axarr[axj, axi].set_yticks(np.linspace(0, 1, 5), minor=False)
            # axarr[axj, axi].yaxis.grid(True, which='minor')
            axarr[axj, axi].margins(y=0.05, x=0.05)
            axarr[axj, axi].autoscale()
            axarr[axj, axi].grid(True)

    # plt.tight_layout()
    f.suptitle("Multi-period correlation Probabilities", fontsize=12)
    # plt.title("Multi-period correlation Probabilities")
    plt.savefig("correlation_probabilities.png")
    plt.show()

"""
