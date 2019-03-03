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

""" This module provides the key correlation matrix classes

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
import statsmodels.multivariate.multivariate_ols as ols
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

from scipy.linalg import lstsq

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html

import matplotlib.pyplot as plt

import correlationMatrix as cm
from correlationMatrix.settings import EIGENVALUE_TOLERANCE


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
            self.matrix = np.asarray(values)
            self.validated = False
        elif type is not None:
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
            # Initialize from file in json format
            q = pd.read_json(json_file)
            self.matrix = np.asarray(q.values)
            self.validated = False
        elif csv_file is not None:
            # Initialize from file in csv format
            q = pd.read_csv(csv_file, index_col=None)
            self.matrix = np.asarray(q.values)
            self.validated = False
        else:
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

    def distance(self):
        """ Compute Euclidean distances implied by a correlation matrix (assuming it is a valid matrix)


        :Example:

        G = A.distance()
        """
        if self.validated:
            distance = np.sqrt(2.0 * (1.0 - self.matrix))
            return distance
        else:
            self.validate()
            if self.validated:
                distance = np.sqrt(2.0 * (1.0 - self.matrix))
                return distance
            else:
                print("Invalid Correlation Matrix")

    def characterize(self):
        """
        TODO Analyse or classify a correlation matrix according to its eigevalue spectrum properties

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
        TODO Create a decomposition of the correlation matrix according to the selected method
        :param method:
        :return:
        """
        pass

    def stress(self, scenario, method):
        """
        TODO Create a stressed correlation matrix according to the selected method
        :param method:
        :return:
        """
        pass

    @property
    def validation_status(self):
        return self.validated


class CorrelationMatrixSet(object):
    """ TODO The _`correlationMatrixSet` object stores a family of correlationMatrix_ objects as a time ordered list. Besides
    storage it allows a variety of simultaneous operations on the collection of matrices


    """
    pass


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

    def fit(self, data, method='pearson'):
        """
        Calculate correlation according to desired measure

        """
        rho = data.corr(method=method).values
        self.matrix = rho

    def pearsonr(self, data):
        # make input1 and input2 uniform
        # x, y = self.make_uniform(dates1, values1, dates2, values2)
        # rho, p = sp.pearsonr(x, y)
        # return rho, p
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

        raw_data1 = self.get_data(input1_url)
        raw_data2 = self.get_data(input2_url)

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
    """  The FactorCorrelationMatrix class fits a variety of factor models

    It stores the derived parameters and modelled correlation matrix values

    TODO compute and store confidence intervals

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ Create a new correlations matrix from sampled data


        """
        # self.samples = kwargs.get('samples')

    def fit(self, data, method='UniformSingleFactor'):
        """
        Estimate a single factor model with uniform loadings
        - The single factor is constructed as the average of all realizations
        - Uniform loadings imply all return realizations are of the same variable r

        """

        # print(data.describe())
        # print(len(data.index))
        # rho, p = sp.pearsonr(data['S4'], F)

        # for i in range(20):
        #         rho, p = sp.pearsonr(Y[:, i], X[:, i])
        #         print(rho)

        # Control that the response variables (r) have the right correlation
        # for i in range(20):
        #     for j in range(20):
        #         rho, p = sp.pearsonr(Y[:, i], Y[:, j])
        #         print(rho)

        if method == 'UniformSingleFactor':
            # Response (dependent) variables
            Y = data.values
            # Control that the response variables (r) have the right correlation
            # rho = data.corr(method='pearson').values
            # print(rho)

            # Compute the row average (Market factor)
            F = data.mean(axis=1).values
            # print(np.std(F0))

            # print(np.std(F1))
            # Normalize the market factor to unit variance
            # Replicate the market factor for the multiple regression
            # X = np.tile(F1, (Y.shape[1], 1)).transpose()
            X = scale(F, with_mean=False, with_std=True)
            # np.reshape(X, (100, 1))
            # np.reshape(X, (-1, 1))

            print(Y.shape)
            print(X.shape)

            corrs = []
            for i in range(Y.shape[1]):
                rho, p = sp.pearsonr(X, Y[:, i])
                corrs.append(rho)
            print(np.mean(corrs) ** 2)
            # res = smf.ols(formula='r ~ F', data=data).fit()
            # estimator = LinearRegression(fit_intercept=False)
            # results = estimator.fit(X, Y)
            # print(dir(results))
            # print(results.coef_)

            b = lstsq(X[:, 0], Y[:, 0])

            # X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            # # y = 1 * x_0 + 2 * x_1 + 3
            # y = np.dot(X, np.array([1, 2])) + 3
            # reg = LinearRegression().fit(X, Y)
            # print(reg.coef_)

            # estimator = ols._MultivariateOLS(X, Y)
            # results = estimator.fit()
            # print(results)
            # plt.plot(b)
            # plt.show()

        # self.matrix = rho

        elif method == 'CAPMModel':
            print("Lets do this")

            raw_data = data.values

            print(raw_data.shape)
            n = raw_data.shape[1] - 1
            # the market factor
            X = raw_data[:, 3]
            # the single stock
            Y = raw_data[:, 0]

            # for i in range(3):
            #     Y = raw_data[:, i]
            #     print(i, sp.pearsonr(Y, X))

            # plt.plot(X, Y)
            # plt.show()

            """
            scipy lstsq API
             
                Parameters:	
                
                a : (M, N) array_like    Left hand side matrix (2-D array).
                b : (M,) or (M, K) array_like  Right hand side matrix or vector (1-D or 2-D array).
                cond : float, optional Cutoff for ‘small’ singular values; used to determine effective rank of a. 
                    Singular values smaller than rcond * largest_singular_value are considered zero.
                overwrite_a : bool, optional Discard data in a (may enhance performance). Default is False.
                overwrite_b : bool, optional Discard data in b (may enhance performance). Default is False.
                check_finite : bool, optional Whether to check that the input matrices contain only finite numbers. 
                    Disabling may give a performance gain, but may result in problems (crashes, non-termination) 
                    if the inputs do contain infinities or NaNs.
                lapack_driver : str, optional Which LAPACK driver is used to solve the least-squares problem. 
                Options are 'gelsd', 'gelsy', 'gelss'. Default ('gelsd') is a good choice. However, 'gelsy' can be slightly faster on many problems. 'gelss' was used historically. It is generally slow but uses less memory.
                                
                Returns:	
                
                x : (N,) or (N, K) ndarray Least-squares solution. Return shape matches shape of b.
                residues : (0,) or () or (K,) ndarray Sums of residues, squared 2-norm for each column in b - a x. 
                    If rank of matrix a is < N or N > M, or 'gelsy' is used, this is a length zero array. If b was 1-D, 
                    this is a () shape array (numpy scalar), otherwise the shape is (K,).
                rank : int  Effective rank of matrix a.
                s : (min(M,N),) ndarray or None  Singular values of a. The condition number of a is abs(s[0] / s[-1]). 
                    None is returned when 'gelsy' is used.

            """

            # TODO stack the individual stock values when the model ask for single loading
            # TODO grouped stock loadings

            # Find all the loadings of entities to the market factor
            a = X
            n = len(a)
            a.shape = (n, 1)
            for i in range(3):
                b = raw_data[:, i]
                p, res, rnk, s = lstsq(a, b)
                print(p, res, rnk, s)

            # # Control that the response variables (r) have the right correlation
            # # rho = data.corr(method='pearson').values
            # # print(rho)
            #
            # # Compute the row average (Market factor)
            # F = data.mean(axis=1).values
            # # print(np.std(F0))
            #
            # # print(np.std(F1))
            # # Normalize the market factor to unit variance
            # # Replicate the market factor for the multiple regression
            # # X = np.tile(F1, (Y.shape[1], 1)).transpose()
            # X = scale(F, with_mean=False, with_std=True)
            # # np.reshape(X, (100, 1))
            # # np.reshape(X, (-1, 1))
            #
            # print(Y.shape)
            # print(X.shape)
            #
            # corrs = []
            # for i in range(Y.shape[1]):
            #     rho, p = sp.pearsonr(X, Y[:, i])
            #     corrs.append(rho)
            # print(np.mean(corrs)**2)

        elif method == 'APTModel':

            raw_data = data.values
            print(raw_data.shape)

            m = 3
            n = raw_data.shape[1]
            a = raw_data[:, n - m:n]
            print(a.shape)

            # b = raw_data[:, 5]
            # n = raw_data.shape[1] - 1
            # # the market factor
            # X = raw_data[:, 3]
            # # the single stock
            # Y = raw_data[:, 0]
            #
            # a = X
            # n = len(a)
            # a.shape = (n, 1)
            for i in range(n - m):
                b = raw_data[:, i]
                p, res, rnk, s = lstsq(a, b)
                print(p, res, rnk, s)

        elif method == 'SectorModel':
            raw_data = data.values
            print('Building Sector Model')

        else:
            print('Invalid Mode')
