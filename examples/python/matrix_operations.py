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
Examples using correlationMatrix to perform various correlation matrix operations.

"""

import numpy as np

import correlationMatrix as cm
from correlationMatrix import dataset_path
from correlationMatrix.utils.converters import matrix_print

print("> Initialize a 3x3 matrix with values")
A = cm.CorrelationMatrix(values=[[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]])
A.print()

print("> Validate that the input matrix satisfies correlation matrix properties")
print(A.validate())

print("> Initialize an invalid matrix")
B = cm.CorrelationMatrix(values=[[-0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
B.print()

print("> Validate whether the input matrix satisfies correlation matrix properties")
print(B.validate())

print("> Initialize a matrix of dimension n without specifying values (Defaults to identity")
C = cm.CorrelationMatrix(dimension=4)
C.print()

print("> Any numpy array can be used for initialization (but not all are valid correlation matrices!)")
D = cm.CorrelationMatrix(values=np.identity(5))
D.print()

print("> All ndarray functionality is available")
E = cm.CorrelationMatrix(values=[[1.0, 0.25], [0.0, 1.0]])
print(E.validate())

print("> Values can be loaded from json or csv files")
F = cm.CorrelationMatrix(json_file=dataset_path + "SingleFactor.json")
F.print()
print(F.validate())

print("> Use pandas style API for saving to files")
A.to_csv("TestMatrix.csv")
A.to_json("TestMatrix.json")

# Obtain the matrix inverse
print("> Derive the inverse correlation matrix")
print(A.inverse())

# Obtain the matrix distance
print("> Derive the distance implied by the correlation matrix")
print(A.distance())

print("> Check that the product of the inverse and correlation matrix is the identify matrix")
print(np.matmul(A.matrix, A.inverse()))

# Generate a random matrix
print("> Generate a random correlation matrix")
G = cm.generate_random_matrix(10)
print(G.validate())

# Apply Cholesky decomposition
print("> Calculate its Cholesky decomposition")
matrix_print(G.decompose('cholesky'), accuracy=2)
