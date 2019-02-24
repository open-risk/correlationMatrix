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

'''
module correlationMatrix.utils - helper classes and functions

'''

from __future__ import print_function, division

import numpy as np
import pandas as pd


def homegenize_dataseries(data):
    pass
    ########## Processes downloaded data ##########

    # Input:
    #   A file containing raw data object (raw_data.R)
    #
    # Transformations:
    #   - Select common dates only
    #   - Select only closing (end of day) values
    #
    # Output:
    #   A csv file with processed data in a dataframe

    # Load data object from file
    # load("raw_data.R")
    # source('SectorsNCompanies.R')
    #
    # # Select only common dates among the different dataseries
    # # start with the first set_
    # base_data_set < - raw_data[[1]]
    # common_dates < - base_data_set$Date
    # print(length(common_dates))
    #
    # k = 1
    # for (i in seq_along(names))
    #     for (j in seq_along(names[[i]]))
    #         {
    #             {
    #                 # for each new set find the common dates
    #                 next_data_set < - raw_data[[k]]
    #         new_common_dates < - intersect(base_data_set$Date, next_data_set$Date)
    #         # keep only common dates for pair
    #         common_dates < - intersect(common_dates, new_common_dates)
    #         print(length(common_dates))
    #         k = k + 1
    #         }
    #         }
    #
    #
    #
    #         # Select only closing (end of day) values
    #         closing_data < - list()
    #         k = 1
    #         for (i in seq_along(vnames))
    #             for (j in seq_along(vnames[[i]]))
    #                 {
    #                     {
    #                         # Select only "closing" value from raw_data
    #                         tmp < - raw_data[[k]][, c("Date", "Adjusted Close")]
    #                 # Select only common dates
    #                 tmp2 < - tmp[tmp$Date % in % common_dates,]
    #                 # Set also the column name
    #                 colnames(tmp2)[2] < - vnames[[i]][j]
    #                 # collect into a list
    #                 closing_data[[k]] < - tmp2[vnames[[i]][j]]
    #                 k = k + 1
    #                 # print(length(closing_data[[i]]$Date))
    #                 }
    #                 }
    #
    #                 # Create data frame from closing data list
    #                 df = data.frame(closing_data)
    #
    #                 # Store closing data to csv file
    #                 write.table(df, file="closing_data.csv", sep=",", row.names = FALSE, col.names = TRUE)
    #


def unique_timestamps(data):
    """
    Identify unique timestamps in a dataframe

    :param data: dataframe. The 'Time' column is used by default

    :returns: returns a numpy array

    """
    unique_timestamps = data['Time'].unique()
    return unique_timestamps


def bin_timestamps(data, cohorts):
    """
    Bin timestamped data in a dataframe so as to have ingoing and outgoing states per cohort interval

    The 'Time' column is used by default

    .. note:: This is a lossy operation: Timestamps are discretised and intermediate state \
    correlations are lost

    """
    # Find the range of observed event times
    t_min = data['Time'].min()
    t_max = data['Time'].max()

    # Divide the range into equal intervals
    dt = (t_max - t_min) / cohorts
    cohort_intervals = [t_min + dt * i for i in range(0, cohorts + 1)]
    sorted_data = data.sort_values(['Time', 'ID'], ascending=[True, True])

    # Identify unique states for validation
    unique_ids = sorted_data['ID'].unique()

    # Arrays to store processed data
    cohort_assigned_state = np.empty((len(unique_ids), len(cohort_intervals)), str)
    cohort_assigned_state.fill(np.nan)
    cohort_event = np.empty((len(unique_ids), len(cohort_intervals)))
    cohort_event.fill(np.nan)
    cohort_count = np.empty((len(unique_ids), len(cohort_intervals)))
    cohort_count.fill(np.nan)

    # Loop over all events and construct a dictionary
    # Create a unique key as per (entity, interval)
    # Add (time, state) pairs as variable length list
    event_dict = {}
    for row in sorted_data.itertuples():
        event_id = row[1]
        event_time = row[2]
        event_state = row[3]
        # Find the interval of the event
        c = int((event_time - event_time % dt) / dt)
        event_key = (event_id, c)
        if event_key in event_dict.keys():
            # append observation if key exists
            event_dict[event_key].append((event_time, event_state))
        else:
            # create new key if not
            event_dict[event_key] = [(event_time, event_state)]

    # Loop over all possible keys
    for i in range(len(unique_ids)):
        for k in range(len(cohort_intervals)):
            event_id = i
            event_cohort = k
            event_key = (i, k)
            # Do we have events in this interval?
            if event_key in event_dict.keys():
                event_list = event_dict[(i, k)]
                # Assign state using last observation in interval
                # TODO Generalize to user specified function
                cohort_assigned_state[event_id, event_cohort] = event_list[len(event_list) - 1][1]
                cohort_event[event_id, event_cohort] = event_list[len(event_list) - 1][0]
                cohort_count[event_id, event_cohort] = int(len(event_list))
                # print('A', cohort_count[event_id, event_cohort])
            elif event_key not in event_dict.keys() and event_cohort > 0:
                # Assign previous state if there are not events and previous state is available
                cohort_assigned_state[event_id, event_cohort] = cohort_assigned_state[event_id, event_cohort - 1]
                cohort_event[event_id, event_cohort] = cohort_event[event_id, event_cohort - 1]
                cohort_count[event_id, event_cohort] = cohort_count[event_id, event_cohort - 1]
                # print('B', cohort_count[event_id, event_cohort])
            elif event_key not in event_dict.keys() and event_cohort == 0:
                # If we don't have observation in first interval assign NaN state
                cohort_assigned_state[event_id, event_cohort] = np.nan
                cohort_event[event_id, event_cohort] = np.nan
                cohort_count[event_id, event_cohort] = np.nan
                # print('C', cohort_count[event_id, event_cohort])

    # Convert to pandas dataframe
    cohort_data = []
    for i in range(len(unique_ids)):
        for c in range(len(cohort_intervals)):
            cohort_data.append((i, c, cohort_assigned_state[i][c], cohort_event[i][c], cohort_count[i][c]))

    cohort_data = pd.DataFrame(cohort_data, columns=['ID', 'Cohort', 'State', 'EventTime', 'Count'])
    return cohort_data, cohort_intervals


def construct_log_returns(data):
    # ########## Construct Index and Log-returns #########
    # #
    # # Input:
    # #   CSV file of stored processed data
    # #
    # # Transformations:
    # # - Create average index
    # # - Calculate log-returns
    # #
    # # Output:
    # #   Log-returns data frame
    #
    # ## Data Cleaning
    #
    # # Clean variables from environment to avoid mistakes
    # # Be careful if you need to save something!!
    #
    # rm(list = ls())
    #
    # # Read names
    # setwd("/home/philippos/Desktop/R_Development/version1.4/")
    # # setwd('/home/philippos/Desktop/R_Development/version1.2.1')
    # source('SectorsNCompanies.R')
    #
    # # Read closing data from csv file
    # df <- read.csv('closing_data.csv', sep=",")
    #
    #
    # average <- rowMeans(df) # calculate global average of all companies
    #
    # sector_average <- list()
    #
    # for (i in seq_along(names))
    #   {
    #       sector_average[[i]] <- rowMeans(df[, vnames[[i]]])
    #   }
    #
    # df2 <- cbind(df,sector_average, average) # creat new data set with average equities as new index
    # # df2 <- cbind(df, average)
    #
    # companies <- list()
    # k = 1
    # for (i in seq_along(vnames))
    #   for (j in seq_along(vnames[[i]]))
    #   {
    #     {
    #       companies[[k]] <- vnames[[i]][j]
    #       k = k + 1
    #     }
    #   }
    #
    #
    # # companiesindex <- c(companies,"S1_FINA", "S2_HLTH", "S3_TECH", "S4_OILG", "S5_CONS", "Index")
    # companiesindex <- c(companies,"S_FINA", "S_HLTH", "S_TECH", "S_OILG", "S_CONS", "Index")
    # # companiesindex <- c(companies, "S1_FINA", "S2_HLTH", "S3_TECH", "S4_OILG", "S5_CONS")
    # # companiesindex <- c(companies, "Index")
    #
    # ## Calculate log-returns of data
    #
    # # calculate log return of all variables
    # log_returns <- list()
    # for (i in seq_along(names(df2))){
    #   log_returns[[i]] <- diff(log(df2[[i]]), lag=1)
    # }
    #
    # # Create data frame from closing data
    # tmp = data.frame(log_returns)
    #
    #
    # # Scale the individual timeseries for zero mean and unit variance
    # ### make it easier to interprate the correlation, we do not care about variance for this application,
    # ### it is not important for the graph, and dependency
    #
    # for (j in seq(1,57)) {
    #   tmp[,j] <- tmp[,j] / sqrt(var(tmp[,j]))
    # }
    # lr <- tmp
    #
    # for (i in seq_along(companiesindex)){
    #   colnames(lr)[i] <- companiesindex[i]
    # }
    #
    # # Store log-return data to csv file
    # # write.table(lr,file="returns_data.csv", sep="," , row.names = FALSE, col.names = TRUE)
    # write.table(lr,file="cleaned_returns_data.csv", sep="," , row.names = FALSE, col.names = TRUE)
    #
    #
    # # check correlations
    # corrs <- cor(lr)
    # corrplot(corrs, tl.cex = 0.5)
    pass
