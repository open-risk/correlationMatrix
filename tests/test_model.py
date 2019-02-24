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


import unittest
import correlationMatrix as cm
from correlationMatrix import source_path
import pandas as pd
from scipy.linalg import expm

ACCURATE_DIGITS = 7


class TestcorrelationMatrix(unittest.TestCase):
    '''
    Default instance (2x2 identity matrix)
    '''
    def test_instantiate_matrix(self):
        a = cm.CorrelationMatrix()
        self.assertAlmostEqual(a[0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[0, 1], 0.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 0], 0.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 1], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)

        b = cm.CorrelationMatrix([[1.0, 3.0], [1.0, 4.0]])
        self.assertAlmostEqual(b[0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b[0, 1], 3.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b[1, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(b[1, 1], 4.0, places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_csv_io(self):
        a = cm.CorrelationMatrix()
        a.to_csv("test.csv")
        b = cm.CorrelationMatrix(csv_file="test.csv")
        self.assertAlmostEqual(a[0, 0], b[0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[0, 1], b[0, 1], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 0], b[1, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 1], b[1, 1], places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_json_io(self):
        a = cm.CorrelationMatrix()
        a.to_json("test.json")
        b = cm.CorrelationMatrix(json_file="test.json")
        self.assertAlmostEqual(a[0, 0], b[0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[0, 1], b[0, 1], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 0], b[1, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 1], b[1, 1], places=ACCURATE_DIGITS, msg=None, delta=None)

    def test_validation(self):
        a = cm.CorrelationMatrix()
        self.assertEqual(a.validate(), True)
        b = cm.CorrelationMatrix(values=[1.0, 3.0])
        self.assertEqual(b.validate()[0][0], 'Matrix Dimensions Differ: ')
        c = cm.CorrelationMatrix(values=[[0.75, 0.25], [0.0, 0.9]])
        self.assertEqual(c.validate()[0][0], 'Rowsum not equal to one: ')
        d = cm.CorrelationMatrix(values=[[0.75, 0.25], [-0.1, 1.1]])
        self.assertEqual(d.validate()[0][0], 'Negative Probabilities: ')

    def test_generator(self):
        a = cm.CorrelationMatrix([[1.0, 3.0], [1.0, 4.0]])
        self.assertAlmostEqual(a[0, 0], expm(a.generator())[0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[0, 1], expm(a.generator())[0, 1], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 0], expm(a.generator())[1, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a[1, 1], expm(a.generator())[1, 1], places=ACCURATE_DIGITS, msg=None, delta=None)


class TestcorrelationMatrixSet(unittest.TestCase):

    def test_instantiate_matrix_set(self):
        periods = 5
        a = cm.CorrelationMatrixSet(dimension=2, periods=periods)
        self.assertEqual(a.temporal_type, 'Incremental')
        self.assertAlmostEqual(a.entries[0][0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        self.assertAlmostEqual(a.entries[periods-1][0, 0], 1.0, places=ACCURATE_DIGITS, msg=None, delta=None)
        pass

    def test_set_validation(self):
        a = cm.CorrelationMatrixSet(dimension=2, periods=5)
        self.assertEqual(a.validate(), True)

    def test_set_cumulate_incremental(self):
        a = cm.CorrelationMatrix(values=[[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        a_set = cm.CorrelationMatrixSet(values=a, periods=3, method='Copy', temporal_type='Incremental')
        b_set = a_set
        b_set.cumulate()
        b_set.incremental()
        self.assertAlmostEqual(a_set.entries[2][0, 0], b_set.entries[2][0, 0], places=ACCURATE_DIGITS, msg=None, delta=None)
        pass

    def test_set_csv_io(self):
        pass

    def test_set_json_io(self):
        pass


class TestStateSpace(unittest.TestCase):

    def test_instantiate_state(self):
        definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                       ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]
        s = cm.StateSpace(definition)
        self.assertEqual(s.definition[0][1], 'AAA')

    def test_get_states(self):
        definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                       ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]
        s = cm.StateSpace(definition)
        self.assertEqual(s.get_states()[0], '0')

    def test_get_state_labels(self):
        definition = [('0', "AAA"), ('1', "AA"), ('2', "A"), ('3', "BBB"),
                       ('4', "BB"), ('5', "B"), ('6', "CCC"), ('7', "D")]
        s = cm.StateSpace(definition)
        self.assertEqual(s.get_state_labels()[0], 'AAA')

    def test_generic(self):
        s = cm.StateSpace()
        n = 10
        s.generic(n=n)
        self.assertEqual(s.get_state_labels()[n-1], str(n-1))

    def test_validate_dataset(self):
        dataset_path = source_path + "datasets/"
        data = pd.read_csv(dataset_path + 'test.csv', dtype={'State': str})
        definition = [('0', "Stage 1"), ('1', "Stage 2"), ('2', "Stage 3")]
        s = cm.StateSpace(definition)
        self.assertEqual(s.validate_dataset(dataset=data)[0], "Dataset contains the expected states.")


if __name__ == "__main__":

    unittest.main()

