
import numpy as np
import matplotlib.pyplot as plt
import TransformationGrid as tg
import gridplotting as gp

# the deformation field 
w = 0.05; # warping parameter, 0-> no warp, 1-> a lot of warp
def phi(x,y):
    return np.array( [x + w* np.sin(2*np.pi*y), \
                      y - w* np.sin(2*np.pi*x)] )

def warp(x2d):
    return phi(x2d[0], x2d[1])

# the jacobian of the deformation field                 
def J_phi(x,y):
    dxdy =  2*np.pi*w*np.cos(2*np.pi*y)
    dydx = -2*np.pi*w*np.cos(2*np.pi*x)
    #TODO: use np.mat() here, see if there's an np.vec()
    return np.array([[1,  dxdy], \
                     [dydx,  1]]) 

# the linear approximation of the deformation field phi near the point a   
# phi(x) ~=  phi(a) + J_phi(a)*(x-a) = J_phi(a)*x + phi(a)-J_phi(a)*a)
a = np.array([0.5, 0.5])
def phi_linear_a(x,y):
    bla = np.array([x-a(1), y-a(2)])
    return phi(a(1),a(2)) +  J_phi(a(1),a(2)).dot(bla)

# 12 x 12 grid of cells (nodes: 13 x 13 )
# on the domain [0,1] x [0, 1]
grid = tg.TransformationGrid(12, 12)
grid_nodes = grid.nodes()
grid_centers = grid.centers()


'''
plt.plot(grid_nodes[:,0],   grid_nodes[:,1], ".")
plt.plot(grid_centers[:,0], grid_centers[:,1], "o")
plt.show()
'''

'''
warped_grid_nodes = np.apply_along_axis(warp, 1, grid_nodes)

x = warped_grid_nodes[:,0]
y = warped_grid_nodes[:,1]
plt.plot(x, y, ".")
plt.show()
'''

def create_local_affine_transform(a2d):
    def T(x2d):
        x_local = [x2d[0]-a2d[0], x2d[1]-a2d[1]]
        bla = J_phi(a2d[0], a2d[1]).dot(x_local)
        return phi(a2d[0], a2d[1]) + bla
    return T


warped_cell_corners_affine = []
for i, j in np.ndindex(grid.num_cells_x, grid.num_cells_y):
    anchor = grid.cellCenter(i,j)
    T = create_local_affine_transform(anchor)
    warped_corners = list(map( lambda c: T(c), grid.cellCorners(i,j) )) 
    warped_cell_corners_affine += [warped_corners]

gp.plot_tiles(np.array(warped_cell_corners_affine))

'''
warped_cell_centers_affine = []
for i, j in np.ndindex(grid.num_cells_x, grid.num_cells_y):
    anchor = grid.cellCenter(i,j)
    T = create_local_affine_transform(anchor)
    
    a = T(anchor)
    warped_cell_centers_affine += [[a[0],a[1]]]


xy = np.array(warped_cell_centers_affine)
# xy = np.apply_along_axis(warp, 1, grid_centers)
x = xy[:,0]
y = xy[:,1]
plt.plot(x, y, ".")
plt.show()
'''