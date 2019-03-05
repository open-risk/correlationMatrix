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

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

import correlationMatrix as cm
from correlationMatrix import source_path
from datasets import Vandermonde

"""
Example workflows using correlationMatrix and matplotlib to generate visualizations of migration phenomena

"""

dataset_path = source_path + "datasets/"
example = 1

if example == 1:
    #
    #  Matplotlib Heatmap Visualization
    #
    myMatrix = cm.CorrelationMatrix(values=Vandermonde)
    myMatrix.print()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = mpl.cm.get_cmap('jet', 30)
    cax = ax1.imshow(myMatrix.matrix, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Vandermonde Correlation')
    fig.colorbar(cax, ticks=[i * 0.1 for i in range(0, 11)])
    # plt.show()
    plt.savefig('vandermonde.png')

elif example == 2:
    #
    #  Scipy Dendrogram Visualization
    #
    G = cm.generate_random_matrix(20)
    Z = linkage(G.distance(), method='single', metric='euclidean', optimal_ordering=False)

    plt.title('Dependency Dendrogram')
    plt.xlabel('Entity')
    plt.ylabel('Correlation Distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        # p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    # plt.show()
    plt.savefig('dendrogram.png')


