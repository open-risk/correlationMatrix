# encoding: utf-8

# (c) 2019-2022 Open Risk (https://www.openriskmanagement.com)
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


""" correlationMatrix -
Python package for statistical analysis and visualization of correlation matrixes """

import os
from .model import *
from .utils import *

__version__ = '0.2'

package_name = 'correlationMatrix'
module_path = os.path.dirname(__file__)
source_path = module_path[:-len(package_name)]
dataset_path = os.path.join(source_path, 'datasets/')

