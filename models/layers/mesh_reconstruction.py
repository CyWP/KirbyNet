from copy import deepcopy

class MeshReconstruction():
    """
    Basically a tree of vertex transformations for rebuilding meshes.
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.roots = []
        self.gemm = deepcopy(mesh.gemm_edges)

    def add_merge(self, edge_id):
       pass 

    class VTransform():
        def __init__(self):
            pass
