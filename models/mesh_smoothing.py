import torch
from torch.nn import ConstantPad2d
import torch.nn.functional as F 
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import numpy as np

class SmoothingModel:
    """Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.mesh = None
        self.loss = None

        # load/define networks
        self.net = networks.define_classifier(
            opt.input_nc,
            opt.ncf,
            opt.ninput_edges,
            0,
            opt,
            self.gpu_ids,
            opt.arch,
            opt.init_type,
            opt.init_gain,
        )
        self.net.train(self.is_train)
        self.criterion = torch.nn.MSELoss().to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data["edge_features"]).float()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(
            self.is_train
        )
        self.mesh = data["mesh"]
        #Debugging to figure out organization of data
        #print(f"Edge features Shape: {self.edge_features.shape}")

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        norm = torch.norm(out)
        inorm = 1/norm
        print("norm", norm)
        self.loss = self.criterion(out*inorm, self.objective(out)*inorm)*norm
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()
    
    def objective(self, out):
        # Compute average of neighbors for every unmasked edge
        obj = 0.2 * out
        for mesh_idx in range(out.shape[0]):
            m = self.mesh[mesh_idx]
            for i in range(4):
                """
                rowval = out[mesh_idx, :, m.gemm_edges[:, i].reshape(-1)]
                if out.shape[-1] != rowval.shape[-1]:
                    rowval = ConstantPad2d((0, out.shape[-1]-rowval.shape[-1], 0, 0), 0)(rowval)
                obj[mesh_idx] += 0.2 * rowval
                """
                rowval = torch.sum(out[mesh_idx, :, m.gemm_edges[:, i].reshape(-1)], dim=0)
                obj[mesh_idx] += 0.2 * F.pad(rowval, (0, obj.shape[-1]-rowval.shape[-1]), mode='constant', value=0)
        #print("diff", obj - out)
        return obj
    ##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = "%s_net.pth" % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print("loading the model from %s" % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = "%s_net.pth" % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            #Create new 'smoothed' mesh 
