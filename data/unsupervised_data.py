import os
import torch
from data.base_dataset import BaseDataset
from util.util import is_mesh_file, pad
from models.layers.mesh import Mesh


class UnsupervisedData(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot)
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.get_mean_std()
        # modify for network later.
        opt.input_nc = self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        mesh = Mesh(
            file=path,
            opt=self.opt,
            hold_history=True,
            export_folder=self.opt.export_folder,
        )
        meta = {"mesh": mesh}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.opt.ninput_edges)
        meta["edge_features"] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size

    def make_dataset(self):
        meshes = []
        for root, _, fnames in os.walk(self.dir): 
            for file in fnames:
                if is_mesh_file(file):
                    meshes.append(os.path.join(root, file))
        return meshes
