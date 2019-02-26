# encoding: utf-8

# (c) 2019 Open Risk (https://www.openriskmanagement.com)
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

""" This module provides the key correlation matrix objects

* correlationMatrix_ implements the functionality of single period correlation matrix
* correlationMatrixSet_ provides a container for a multiperiod correlation matrix collection
* EmpiricalCorrelationMatrix implements the functionality of a continuously observed correlation matrix

"""

import json
import os

import numpy as np
import pandas as pd
import requests
import scipy.stats as sp
from scipy.linalg import eigh
from scipy.linalg import inv

import correlationMatrix as cm
from correlationMatrix.settings import EIGENVALUE_TOLERANCE


class CorrelationMatrix:
    # class CorrelationMatrix(np.matrix):
    """ The _`correlationMatrix` object implements a typical (one period) `correlation matrix <https://www.openriskmanual.org/wiki/correlation_Matrix>`_.
    The class inherits from numpy ndarray (instead of matrix because the latter will be deprecated

    It implements additional properties specific to correlation matrices.

    It forms the building block of the correlationMatrixSet_ which holds a collection of matrices
    in increasing temporal order (to capture systems with time varying correlation)

    This class does not implement any estimation method, this task is relegated to classes
    EmpiricalCorrelationMatrix -> Full empirical estimation
    FactorCorrelationMatrix -> Factor Model estimation
    PCAMatrix -> PCA Model estimation

    """

    def __init__(self, values=None, type=None, json_file=None, csv_file=None, **kwargs):

        """ Create a new correlation matrix.

        Different options for initialization are:

        * providing values as a list of list
        * providing values as a numpy array
        * indicating matrix type and additional parameters
        * loading values from a csv file
        * loading values from a json file

        The above initializations are mutually exclusive and are tested until a valid option is
        found. Without a valid option, a default identity matrix is generated

        :param values: initialization values
        :param type: matrix dimensionality (default is 2)
        :param json_file: a json file containing correlation matrix data
        :param csv_file: a csv file containing correlation matrix data
        :type values: list of lists or numpy array
        :type type: string
        :returns: returns a correlationMatrix object
        :rtype: object

        .. note:: The initialization in itself does not validate if the provided values form indeed a correlation matrix

        :Example:

        .. code-block:: python

            A = cm.correlationMatrix(values=[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])

        """

        if values is not None:
            # Initialize with given values
            print('given values', values)
            self.matrix = np.asarray(values)
            self.validated = False
        elif type is not None:
            print('given type')
            self.matrix = np.identity(2)
            if type == 'UniformSingleFactor':
                rho = kwargs.get('rho')
                n = kwargs.get('n')
                tmp1 = np.identity(n)
                tmp2 = rho * (np.tri(n) - tmp1)
                tmp3 = tmp2.transpose()
                self.matrix = np.asarray(tmp1 + tmp2 + tmp3)
            elif type == 'TBD':
                pass
            else:
                pass
            # validation flag is set to True for modelled Matrices
            self.validated = True
        elif json_file is not None:
            print('given file')
            # Initialize from file in json format
            q = pd.read_json(json_file)
            self.matrix = np.asarray(q.values)
            self.validated = False
        elif csv_file is not None:
            print('given file')
            # Initialize from file in csv format
            q = pd.read_csv(csv_file, index_col=None)
            self.matrix = np.asarray(q.values)
            self.validated = False
        else:
            print('no input')
            # Default instance (2x2 identity matrix)
            default = np.identity(2)
            self.matrix = np.asarray(default)
            self.validated = False

        # temporary dimension assignment (must validated for squareness)
        self.dimension = self.matrix.shape[0]

    def to_json(self, file):
        """
        Write correlation matrix to file in json format

        :param file: json filename
        """

        q = pd.DataFrame(self.matrix)
        q.to_json(file, orient='values')

    def to_csv(self, file):
        """
        Write correlation matrix to file in csv format

        :param file: csv filename
        """

        q = pd.DataFrame(self.matrix)
        q.to_csv(file, index=None)

    def to_html(self, file=None):
        html_table = pd.DataFrame(self).to_html()
        if file is not None:
            file = open(file, 'w')
            file.write(html_table)
            file.close()
        return html_table

    def fix_negative_values(self):
        """
        If a matrix entity is below zero, set to zero and correct the diagonal element to enforce

        """

        matrix = self.matrix
        matrix_size = matrix.shape[0]
        # For all rows
        # Search all cols for negative entries
        for i in range(matrix_size):
            for j in range(matrix_size):
                if matrix[i, j] < 0.0:
                    self.matrix[i, j] = 0.0

    def validate(self, accuracy=1e-3):
        """ Validate required properties of an input correlation matrix. The following are checked

        1. check squareness
        2. check symmetry
        3. check that all values are between -1 and 1 and diagonal elements == 1
        4. check positive definiteness

        :param accuracy: accuracy level to use for validation
        :type accuracy: float

        :returns: List of tuples with validation messages
        """
        validation_messages = []

        matrix = self.matrix
        # checking squareness of matrix
        if matrix.shape[0] != matrix.shape[1]:
            validation_messages.append(("Matrix Dimensions Differ: ", matrix.shape))
        else:
            matrix_size = matrix.shape[0]
            # checking that values of matrix are within allowed range
            for i in range(matrix_size):
                if matrix[i, i] != 1:
                    validation_messages.append(("Diagonal Values different than 1: ", (i, matrix[i, i])))
                for j in range(matrix_size):
                    if matrix[i, j] < -1:
                        validation_messages.append(("Values less than -1: ", (i, j, matrix[i, j])))
                    if matrix[i, j] > 1:
                        validation_messages.append(("Values larger than 1: ", (i, j, matrix[i, j])))
            # checking symmetry
            for i in range(matrix_size):
                for j in range(matrix_size):
                    if matrix[i, j] != matrix[j, i]:
                        validation_messages.append(("Symmetry violating value: ", (i, j, matrix[i, j])))
            # checking positive semi-definitess (non-negative eigenvalues)
            Eigenvalues, Decomposition = eigh(matrix)
            if not np.all(Eigenvalues > - EIGENVALUE_TOLERANCE):
                validation_messages.append(("Matrix is not positive semi-definite"))

        if len(validation_messages) == 0:
            self.validated = True
            self.dimension = matrix.shape[0]
            return self.validated
        else:
            self.validated = False
            return validation_messages

    def inverse(self):
        """ Compute the inverse of a correlation matrix (assuming it is a valid matrix)


        :Example:

        G = A.inverse()
        """
        if self.validated:
            inverse = inv(self.matrix)
            return inverse
        else:
            self.validate()
            if self.validated:
                inverse = inv(self.matrix)
                return inverse
            else:
                print("Invalid Correlation Matrix")

    def characterize(self):
        """ Analyse or classify a correlation matrix according to its properties

        * diagonal dominance

        .. Todo:: Further characterization
        """

        outcome_messages = []
        if self.validated is True:
            matrix = self.matrix
            matrix_size = matrix.shape[0]
            dominance = True
            for i in range(matrix_size):
                if matrix[i, i] < 0.5:
                    dominance = False
            if dominance:
                outcome_messages.append("Diagonally Dominant")
            else:
                outcome_messages.append("Not Diagonally Dominant")
        else:
            outcome_messages.append("Not a validated matrix. Use matrix.validate()")
        return outcome_messages

    def generate_random_matrix(self):
        """

        .. Todo:: Implement matrix generation subject to various constraints
        """
        pass

    def print(self, format_type='Standard', accuracy=2):
        """ Pretty print a correlation matrix

        :param format_type: formatting options (Standard, Percent)
        :type format_type: str
        :param accuracy: number of decimals to display
        :type accuracy: int

        """
        for s_in in range(self.matrix.shape[0]):
            for s_out in range(self.matrix.shape[1]):
                if format_type is 'Standard':
                    format_string = "{0:." + str(accuracy) + "f}"
                    print(format_string.format(self.matrix[s_in, s_out]) + ' ', end='')
                elif format_type is 'Percent':
                    print("{0:.2f}%".format(100 * self.matrix[s_in, s_out]) + ' ', end='')
            print('')
        print('')

    def decompose(self, method):
        """
        Create a decomposition of the correlation matrix according to the selected method
        :param method:
        :return:
        """
        pass

    @property
    def validation_status(self):
        return self.validated


class CorrelationMatrixSet(object):
    """  The _`correlationMatrixSet` object stores a family of correlationMatrix_ objects as a time ordered list. Besides
    storage it allows a variety of simultaneous operations on the collection of matrices


    """

    def __init__(self, dimension=2, values=None, periods=1, temporal_type=None, method=None, json_file=None,
                 csv_file=None):
        """ Create a new matrix set. Different options for initialization are:

        * providing values as a list of list
        * providing values as a numpy array
        * loading from a csv file
        * loading from a json file

        Without data, a default identity matrix is generated with user specified dimension

        :param values: initialization values
        :param dimension: matrix dimensionality (default is 2)
        :param method: matrix dimensionality (default is 2)
        :param periods: List with the timesteps of matrix observations
        :param temporal_type: matrix dimensionality (default is 2)

        * Incremental: Each period matrix reflects correlations for that period
        * Cumulative: Each period matrix reflects cumulative correlations from start to that period

        :param json_file: a json file containing correlation matrix data
        :param csv_file: a csv file containing correlation matrix data

        :type values: list of lists or numpy array
        :type dimension: int
        :type temporal_type: str
        :type json_file: str
        :type csv_file: str

        :returns: returns a TranstionMatrix object
        :rtype: object

        .. note:: The initialization in itself does not validate if the provided values form indeed a correlation matrix set

        :Example:

        Instantiate a correlation matrix set directly using a list of matrices

        .. code-block:: python

            C_Vals = [[[0.75, 0.25], [0.0, 1.0]],  [[0.75, 0.25], [0.0, 1.0]]]

            C_Set = cm.correlationMatrixSet(values=C_Vals, temporal_type='Incremental')

        """

        if values is not None:
            # Copy a single matrix to all periods
            if method is 'Copy':
                val_set = []
                for k in range(periods):
                    a = cm.CorrelationMatrix(values)
                    val_set.append(a)
                self.entries = val_set
                self.temporal_type = 'Incremental'
                self.periods = list(range(periods))
                self.dimension = val_set[0].shape[0]
            # Create a multi-period matrix assuming a Markov Chain
            elif method is 'Power':
                val_set = []
                a = cm.CorrelationMatrix(values)
                val_set.append(a)
                an = a
                for k in range(periods - 1):
                    an = an * a
                    an = cm.CorrelationMatrix(an)
                    val_set.append(an)
                self.entries = val_set
                self.temporal_type = 'Cumulative'
                self.periods = list(range(periods))
                self.dimension = val_set[0].shape[0]
            # Use provided matrices as-is
            elif method is None:
                val_set = []
                for entry in values:
                    a = cm.CorrelationMatrix(entry)
                    val_set.append(a)
                self.entries = val_set
                self.temporal_type = temporal_type
                self.periods = list(range(periods))
                self.dimension = val_set[0].shape[0]
        elif values is None and csv_file is not None:
            # Initialize from file in csv format
            # First row is meta data labels (From States, To States, Periods, Tenor List)
            # Second row is meta data values (comma separated)
            # Subsequent rows are Periods x Matrices in sequence
            if not os.path.isfile(csv_file):
                print("Input File Does not Exist")
                exit()
            f = open(csv_file)
            header_dict = f.readline()
            header_data = f.readline().split(',')
            val_set = []
            from_states = int(header_data.pop(0))
            to_states = int(header_data.pop(0))
            periods = int(header_data.pop(0))
            tenors = [int(x) for x in header_data]
            q = pd.read_csv(f, header=None, usecols=range(to_states))
            for k in range(periods):
                raw = q.iloc[k * from_states:(k + 1) * from_states]
                a = cm.CorrelationMatrix(raw.as_matrix())
                val_set.append(a)
            self.entries = val_set
            self.temporal_type = temporal_type
            self.periods = tenors
            self.dimension = val_set[0].shape[0]
            f.close()
        elif values is None and json_file is not None:
            # Initialize from file in json format
            if not os.path.isfile(json_file):
                print("Input File Does not Exist")
                exit()
            val_set = []
            q = json.load(open(json_file))
            periods = len(q)
            for k in range(periods):
                a = cm.CorrelationMatrix(q[k])
                val_set.append(a)
            self.entries = val_set
            self.temporal_type = temporal_type
            self.periods = list(range(periods))
            self.dimension = val_set[0].shape[0]
        else:
            # Default instance (2x2 identity matrix)
            # default = np.identity(dimension)
            val_set = []
            for k in range(periods):
                a = cm.CorrelationMatrix(dimension=dimension)
                val_set.append(a)
            self.entries = val_set
            if temporal_type is not None:
                self.temporal_type = temporal_type
            else:
                self.temporal_type = 'Incremental'
            self.periods = list(range(periods))
            self.dimension = 2

        self.validated = False
        return

    def __mul__(self, scale):
        """ Scale all entries of the set by a factor


        """
        scaled = self
        val_set = []
        for entry in self.entries:
            a = entry * scale
            val_set.append(a)
        scaled.entries = val_set
        return scaled

    def validate(self):
        """ Validate correlation matrix set (validating individual entries)

        :returns: List of tuples with validation messages
        """
        validation_messages = []
        validation_status = []
        for entry in self.entries:
            validation_messages.append(entry.validate())
            validation_status.append(entry.validated)
        if all(validation_status):
            self.validated = True
            return self.validated
        else:
            self.validated = False
            return validation_messages

    def remove(self, state, method):
        """ remove a correlation matrix state and distribute its probability to other states according
        to prescribed method

        """
        updated = self
        val_set = []
        for entry in self.entries:
            a = entry.remove(state, method)
            val_set.append(a)
        updated.entries = val_set
        return updated

    def print_matrix(self, format_type='Standard', accuracy=2):
        """ Pretty Print a correlation Matrix Set

        """
        k = 0
        for entry in self.entries:
            print("Entry: ", k)
            entry.print(format_type=format_type, accuracy=accuracy)
            k += 1

    def to_json(self, file=None, accuracy=5):
        hold = []
        for k in range(len(self.entries)):
            entry = np.around(self.entries[k], accuracy)
            hold.append(entry.tolist())
        serialized = json.dumps(hold, indent=2, separators=(',', ': '))
        if file is not None:
            file = open(file, 'w')
            file.write(serialized)
            file.close()

        return serialized

    def to_csv(self, file):
        pass

    def to_html(self, file=None):
        table_set = ''
        for table in self.entries:
            html_table = pd.DataFrame(table).to_html()
            table_set += html_table
        if file is not None:
            file = open(file, 'w')
            file.write(table_set)
            file.close()
        return table_set


class EmpiricalCorrelationMatrix(CorrelationMatrix):
    """  The EmpiricalCorrelationMatrix object stores the full empirical correlation Matrix.

    It stores matrices estimated using any of the standard correlation metrics
    (Pearson, Kendal, Tau)


    """

    def __init__(self, **kwargs):

        super().__init__(values=None, type=None, json_file=None, csv_file=None, **kwargs)

        """ Create a new correlations matrix from sampled data
        

        """
        # self.samples = kwargs.get('samples')

    def get_data(self, data_url):
        r = requests.get(data_url)
        return r.json()

    def make_uniform(self, dates1, values1, dates2, values2):
        # make the two timeseries arrays uniform (select common observation dates)
        # find common dates
        # return values on common dates
        common_dates = list(set(dates1).intersection(dates2))

        new_values1 = []
        new_values2 = []
        for date in common_dates:
            i1 = dates1.index(date)
            i2 = dates2.index(date)
            new_values1.append(values1[i1])
            new_values2.append(values2[i2])

        x = new_values1
        y = new_values2
        return x, y

    # calculate the pearson correlation
    def pearsonr(self, data):
        # make input1 and input2 uniform
        # x, y = self.make_uniform(dates1, values1, dates2, values2)
        # rho, p = sp.pearsonr(x, y)
        # return rho, p
        print('Inside', type(self))
        rho = data.corr(method='pearson').values
        self.matrix = rho

    # calculate the kendall correlation
    def kendallr(self, dates1, values1, dates2, values2):
        # make input1 and input2 uniform
        x, y = self.make_uniform(dates1, values1, dates2, values2)
        # print(x, y)
        rho, p = sp.kendalltau(x, y)
        return rho, p

    # calculate the spearman correlation
    def spearmanr(self, dates1, values1, dates2, values2):
        # make input1 and input2 uniform
        x, y = self.make_uniform(dates1, values1, dates2, values2)
        rho, p = sp.spearmanr(x, y)
        return rho, p

    # the correlation collection (ADD OTHER functions)
    def calculate_correlation(self, model_name, input1_url, input2_url):
        # python models evaluated directly
        # c++ models evaluated via CGI requests

        # print(model_name)
        # Exctract timeseries data for calculation

        raw_data1 = get_data(input1_url)
        raw_data2 = get_data(input2_url)

        json_string1 = raw_data1['_items'][0]['json_dump']
        Data1 = json.loads(json_string1)
        dates1 = Data1['Dates']
        values1 = Data1['Values']
        # print(dates1, values1)

        json_string2 = raw_data2['_items'][0]['json_dump']
        Data2 = json.loads(json_string2)
        dates2 = Data2['Dates']
        values2 = Data2['Values']

        if model_name == 'Pearson_Correlation':
            rho, p = self.pearsonr(dates1, values1, dates2, values2)
        elif model_name == 'Kendall_Correlation':
            rho, p = self.kendallr(dates1, values1, dates2, values2)
        elif model_name == 'Spearman_Correlation':
            rho, p = self.spearmanr(dates1, values1, dates2, values2)
        return {'rho': rho, 'p': p}


class FactorCorrelationMatrix(CorrelationMatrix):
    """  The EmpiricalCorrelationMatrix object stores the full empirical correlation Matrix.

    It stores matrices estimated using any of the standard correlation metrics
    (Pearson, Kendal, Tau)

    The EmpiricalCorrelationMatrix object is different from the correlationMatrixSet in that it stores detailed event time
    of observations and the correlation densities in addition to the correlation probabilities

    An EmpiricalCorrelationMatrix can be converted into a correlationMatrixSet by sampling on a temporal grid (but not
    vice-versa)


    """

    def __init__(self, dimension=2, values=None, observation_times=None, json_file=None,
                 csv_file=None):
        CorrelationMatrix.__init__(self)
        """ Create a new probability matrix. Different options for initialization are:

        * providing values as a 3D numpy array of signature (S, S, T) and observation times as a list or numpy array of length T
        * loading from a csv file
        * loading from a json file

        Without data, a default identity matrix is generated with user specified dimension

        :param values: initialization values
        :param dimension: matrix dimensionality (default is 2)
        :param observation_times: List with the timesteps (support) of correlation observations
        :param json_file: a json file containing correlation matrix data
        :param csv_file: a csv file containing correlation matrix data

        :type values: 3D numpy array
        :type dimension: int
        :type observations: int
        :type json_file: str
        :type csv_file: str

        :returns: returns a EmpiricalCorrelationMatrix object
        :rtype: object

        .. note:: The initialization in itself does not validate if the provided values form indeed a correlation matrix set

        :Example:

        Instantiate a correlation probability matrix

        .. code-block:: python

        """

        self.values = values
        self.observation_times = observation_times

        return

    def get_data(self, data_url):
        r = requests.get(data_url)
        return r.json()

    def make_uniform(self, dates1, values1, dates2, values2):
        # make the two timeseries arrays uniform (select common observation dates)
        # find common dates
        # return values on common dates
        common_dates = list(set(dates1).intersection(dates2))

        new_values1 = []
        new_values2 = []
        for date in common_dates:
            i1 = dates1.index(date)
            i2 = dates2.index(date)
            new_values1.append(values1[i1])
            new_values2.append(values2[i2])

        x = new_values1
        y = new_values2
        return x, y

    # inputs to the library are assumed to be

    # calculate the pearson correlation
    def pearsonr(self, dates1, values1, dates2, values2):
        # make input1 and input2 uniform
        x, y = self.make_uniform(dates1, values1, dates2, values2)
        rho, p = sp.pearsonr(x, y)
        return rho, p

    # calculate the kendall correlation
    def kendallr(self, dates1, values1, dates2, values2):
        # make input1 and input2 uniform
        x, y = self.make_uniform(dates1, values1, dates2, values2)
        # print(x, y)
        rho, p = sp.kendalltau(x, y)
        return rho, p

    # calculate the spearman correlation
    def spearmanr(self, dates1, values1, dates2, values2):
        # make input1 and input2 uniform
        x, y = self.make_uniform(dates1, values1, dates2, values2)
        rho, p = sp.spearmanr(x, y)
        return rho, p

    # the correlation collection (ADD OTHER functions)
    def calculate_correlation(self, model_name, input1_url, input2_url):
        # python models evaluated directly
        # c++ models evaluated via CGI requests

        # print(model_name)
        # Exctract timeseries data for calculation

        raw_data1 = get_data(input1_url)
        raw_data2 = get_data(input2_url)

        json_string1 = raw_data1['_items'][0]['json_dump']
        Data1 = json.loads(json_string1)
        dates1 = Data1['Dates']
        values1 = Data1['Values']
        # print(dates1, values1)

        json_string2 = raw_data2['_items'][0]['json_dump']
        Data2 = json.loads(json_string2)
        dates2 = Data2['Dates']
        values2 = Data2['Values']

        if model_name == 'Pearson_Correlation':
            rho, p = self.pearsonr(dates1, values1, dates2, values2)
        elif model_name == 'Kendall_Correlation':
            rho, p = self.kendallr(dates1, values1, dates2, values2)
        elif model_name == 'Spearman_Correlation':
            rho, p = self.spearmanr(dates1, values1, dates2, values2)
        return {'rho': rho, 'p': p}

    def hierarchical(self):
        ########## Fit Hierachical Factor Model ##########
        # we need separate the residuals, if the residuals are correlated with each other, we won't make the garphs.
        # Basically, in the kiwi paper, he express the residuals only have one sector, but we have five sectors
        # Firstly we do the sector model, sector indics are just the averages of companies within each sector
        #
        # Input:
        #   CSV file of scaled log-return data
        #
        # Output:
        #   Linear factor model and residuals
        # rm(list=ls())
        # library(corrplot)
        # # Read closing data from csv file
        # setwd("C:\\Users\\lixua\\Desktop\\version1.4")
        # # setwd('/home/philippos/Desktop/R_Development/version1.2.1')
        # source('SectorsNCompanies.R')
        # # setwd('/home/philippos/Desktop/R_Development/Current')
        # df < - read.csv('cleaned_returns_data.csv', sep=",")
        #
        # ### Calculate the Sector Loadings on the Index and the Sector Residuals ###
        #
        # sector_fit < - lm(data.matrix(df[, 51:55])
        # ~ Index, data = df)
        # s_load < - sector_fit$coefficients
        # s_res < - data.frame(sector_fit$residuals)
        # s_corr < - cor(s_res)
        # corrplot(s_corr)
        #
        # ### Calculate company loadings on the Index and the company Residuals ###
        #
        # df2 < - data.frame(df[, 1:50], s_res, df["Index"])
        #
        # company_fit < - lm(data.matrix(df2[, 1:50])
        # ~ Index + S_FINA + S_HLTH + S_TECH + S_OILG + S_CONS, data = df2)
        # c_load < - company_fit$coefficients
        # c_res < - data.frame(company_fit$residuals)
        # c_corr < - cor(c_res)
        # corrplot(c_corr, tl.cex = 0.5)
        #
        #
        # ### Store Sector and Company Residuales ###
        # write.table(s_res, file="sector_residuals.csv", sep=",", row.names = FALSE, col.names = TRUE)
        # write.table(c_res, file="company_residuals.csv", sep=",", row.names = FALSE, col.names = TRUE)
        pass
