import numpy as np

class TransformationGrid:
    def __init__(self, num_cells_x, num_cells_y):
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.transforms = [[np.eye(3)] * num_cells_y] * num_cells_x
        self.domain_x = [0, 1] 
        self.domain_y = [0, 1] 
    
    def cellCorners(self, i, j):
        assert (i>=0 and i<self.num_cells_x)
        assert (j>=0 and i<self.num_cells_y)

        cell_width = self.cellWidth()
        cell_height = self.cellHeight()

        return [ [i*cell_width, j*cell_height], \
                 [(i+1)*cell_width, j*cell_height], \
                 [(i+1)*cell_width, (j+1)*cell_height], \
                 [i*cell_width, (j+1)*cell_height] \
               ]
    def cellCenter(self, i, j):
        cw = self.cellWidth()
        ch = self.cellHeight()
        return [(i+0.5)*cw, (j+0.5)*ch]

    def cellWidth(self):
        return (self.domain_x[1]-self.domain_x[0])/self.num_cells_x

    def cellHeight(self):
        return (self.domain_y[1]-self.domain_y[0])/self.num_cells_y

    def nodes(self):
        Nx = self.num_cells_x
        Ny = self.num_cells_y
        cw = self.cellWidth()
        ch = self.cellHeight()
        left = self.domain_x[0]
        top = self.domain_y[0]
        gnx = [ left + i*cw for i in range(0,Nx+1) ]
        gny = [ top + j*ch for j in range(0,Ny+1) ]
        return np.transpose([np.tile(gnx, len(gny)), np.repeat(gny, len(gnx))])    

    def centers(self):
        Nx = self.num_cells_x
        Ny = self.num_cells_y
        cw = self.cellWidth()
        ch = self.cellHeight()
        left = self.domain_x[0]
        top = self.domain_y[0]
        gcx = [ left + (i+0.5)*cw for i in range(0,Nx) ]
        gcy = [ top + (j+0.5)*ch for j in range(0,Ny) ]
        return np.transpose([np.tile(gcx, len(gcy)), np.repeat(gcy, len(gcx))])