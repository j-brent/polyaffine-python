import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def plot_tiles(corners_collection):
    
    patches = []
    for corners in corners_collection:
        patches.append(Polygon(corners, closed=True))

    fig, ax = plt.subplots()
    p = PatchCollection(patches, alpha=0.2, edgeColors='blue')
    #colors = 100*np.random.rand(len(patches))
    #p.set_array(np.array(colors))
    ax.add_collection(p)
    #fig.colorbar(p, ax=ax)

    plt.show()
