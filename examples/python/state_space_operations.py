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
Examples using correlationMatrix to perform various state space operations

"""

import numpy as np


import correlationMatrix as cm
from datasets import JLT
from correlationMatrix import dataset_path

myState = cm.SnP_SS
print(myState.get_states())
print(myState.get_state_labels())
print(myState.definition)

