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

"""
module correlationMatrix.utils - helper classes and functions

"""

import numpy as np
import pandas as pd


def csv_files_to_frame(list, directory, filename):
    """ Given a list of symbols with timeseries data

        - iterate through a directory for csv files
        - load and merge file data into a single dataframe

    """

    df = pd.DataFrame()
    for entry in list:
        entry_name = entry[2]
        entity_file = entry_name + ".csv"
        entity_data = pd.read_csv(directory + entity_file)
        select_data = entity_data.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume'])
        index_data = select_data.set_index('Date')
        rename_data = index_data.rename(columns={"Adj Close": entry_name})
        df = pd.concat([df, rename_data], axis=1, sort=False)

    df.to_csv(filename)

    return df


def json_file_to_frame(input_filename, output_filename):
    """ Given a file name with json data in the format

        .. code:: python

            {
                "Entity1" : [Values],
                "Entity2" : [Values],
                ...
                "EntityN" : [Values]
            }

    Convert the data to a pandas dataframe for further processing

    """

    entity_data = pd.read_json(open(input_filename, mode='r'))
    entity_data.to_csv(output_filename, index=False)

    return entity_data


def construct_log_returns(in_filename, out_filename, drop_columns=None):
    """
        Load a dataframe with level data from file
        Drop a list of columns that are not to be processed
        Construct log-returns
        Store to file

    """
    if drop_columns:
        level_data = pd.read_csv(in_filename).drop(columns=drop_columns)
    else:
        level_data = pd.read_csv(in_filename)

    log_return_data = pd.DataFrame()

    for column in level_data:
        log_return_data[column] = np.log(level_data[column]) - np.log(level_data[column].shift(1))

    log_return_data = log_return_data.dropna()
    log_return_data.to_csv(out_filename, index=False)


def construct_returns(in_filename, out_filename, drop_columns=None):
    """
        Load a dataframe with level data from file
        Drop a list of columns that are not to be processed
        Construct simple returns
        Store to file

    """
    if drop_columns:
        level_data = pd.read_csv(in_filename).drop(columns=drop_columns)
    else:
        level_data = pd.read_csv(in_filename)

    return_data = pd.DataFrame()

    for column in level_data:
        return_data[column] = level_data[column] - level_data[column].shift(1)

    return_data = return_data.dropna()
    return_data.to_csv(out_filename, index=False)


def normalize_log_returns(in_filename, out_filename):
    """
        Load a dataframe with log-return data from file
        Normalize to zero mean and unit variance
        Store to file

    """
    mean_vals, std_vals = [], []
    log_return_data = pd.read_csv(in_filename)
    data = log_return_data.values
    cols = list(log_return_data)
    print('Entity Names: ', cols)
    scaled_data = np.asarray(data)
    for ts in range(data.shape[1]):
        mean = data[:, ts].mean()
        std = data[:, ts].std()
        scaled_data[:, ts] = (data[:, ts] - mean) / std

    scaled_returns = pd.DataFrame(scaled_data, columns=cols)
    scaled_returns.to_csv(out_filename, index=False)

    for ts in range(data.shape[1]):
        mean_vals.append(scaled_data[:, ts].mean())
        std_vals.append(scaled_data[:, ts].std())

    return mean_vals, std_vals
