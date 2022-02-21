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


import unittest

import correlationMatrix as cm

ACCURATE_DIGITS = 7


class TestcorrelationMatrix(unittest.TestCase):
    '''
    Default instance (2x2 identity matrix)
    '''

    def test_instantiate_matrix(self):
        a = cm.CorrelationMatrix()
        self.assertAlmostEqual(a.matrix[0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[0, 1], 0.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 0], 0.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 1], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)

        b = cm.CorrelationMatrix([[1.0, 3.0], [1.0, 4.0]])
        self.assertAlmostEqual(b.matrix[0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b.matrix[0, 1], 3.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b.matrix[1, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b.matrix[1, 1], 4.0, places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_csv_io(self):
        a = cm.CorrelationMatrix()
        a.to_csv("test.csv")
        b = cm.CorrelationMatrix(csv_file="test.csv")
        self.assertAlmostEqual(a.matrix[0, 0], b.matrix[0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[0, 1], b.matrix[0, 1], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 0], b.matrix[1, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 1], b.matrix[1, 1], places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_json_io(self):
        a = cm.CorrelationMatrix()
        a.to_json("test.json")
        b = cm.CorrelationMatrix(json_file="test.json")
        self.assertAlmostEqual(a.matrix[0, 0], b.matrix[0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[0, 1], b.matrix[0, 1], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 0], b.matrix[1, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.matrix[1, 1], b.matrix[1, 1], places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_validation(self):
        a = cm.CorrelationMatrix()
        self.assertEqual(a.validate(), True)
        b = cm.CorrelationMatrix(values=[1.0, 3.0])
        self.assertEqual(b.validate()[0][0], 'Matrix Non Square: ')
        c = cm.CorrelationMatrix(values=[[0.75, 0.25], [0.0, 0.9]])
        self.assertEqual(c.validate()[0][0], 'Diagonal Values different than 1: ')
        d = cm.CorrelationMatrix(values=[[1.0, 0.1], [-0.1, 1.0]])
        self.assertEqual(d.validate()[0][0], 'Symmetry violating value: ')


if __name__ == "__main__":
    unittest.main()
