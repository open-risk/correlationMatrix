# encoding: utf-8

# (c) 2019-2024 Open Risk, all rights reserved
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
from datasets import Minimal
from correlationMatrix import source_path, dataset_path

ACCURATE_DIGITS = 7


class TestDatasets(unittest.TestCase):
    """
    Load in-memory matrices
    """

    def test_minimal_matrix(self):
        a = cm.CorrelationMatrix(values=Minimal)
        a.validate()
        self.assertEqual(a.dimension, 3)


if __name__ == "__main__":
    unittest.main()
