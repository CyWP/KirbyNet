import torch
from torch.nn import ConstantPad2d
import torch.nn.functional as F 
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network
import numpy as np

class SmoothingModel:
    """
    Class for training Model weights
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.mean = self.std = None
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
        self.energy = self.out_var = self.out_norm = self.out_mean = None

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

    def get_training_stats(self):
        return {'norm':self.out_norm, 'energy': self.energy, 'mean': self.out_mean, 'var': self.out_var}
    
    def get_mesh_energy(self):
        if self.mesh is None:
            return None
        return np.array([m.mean_curvature_energy() for m in self.mesh])

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        norm = torch.norm(out).item()
        inorm = 1/norm
        self.energy = self.get_mesh_energy()
        self.out_norm = norm
        self.out_mean = torch.mean(out**2).item()
        self.out_var = out.var().item()
        obj = self.objective(out)
        self.loss = self.criterion(out, obj)+0.005/self.out_mean
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
                rowval = torch.sum(out[mesh_idx, :, m.gemm_edges[:, i].reshape(-1)], dim=0)
                obj[mesh_idx] += 0.2 * F.pad(rowval, (0, obj.shape[-1]-rowval.shape[-1]), mode='constant', value=0)
        return obj
    ##################
    
    def set_mean_std(self, mean, std):
        self.mean = mean
        self.std = std

    def load_network(self, which_epoch):
        """Load model and options from disk."""
        save_filename = "%s_net.pth" % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print("Loading the model and options from %s" % load_path)
        checkpoint = torch.load(load_path, map_location=str(self.device))
        state_dict = checkpoint['model_state_dict']
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)
        opt = checkpoint.get('opt', {})

    def save_network(self, which_epoch):
        """Save model and options to disk."""
        save_filename = "%s_net.pth" % which_epoch
        save_path = join(self.save_dir, save_filename)
        opt_dict = vars(self.opt)
        for k, v in opt_dict.items():
            if isinstance(v, np.ndarray):
                opt_dict[k] = v.tolist()
            mean = self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean
            std = self.std.tolist() if isinstance(self.std, np.ndarray) else self.std
        checkpoint = {
            'model_state_dict': self.net.module.cpu().state_dict() if len(self.gpu_ids) > 0 and torch.cuda.is_available() else self.net.cpu().state_dict(),
            'opt': opt_dict,
            'mean': mean,
            'std': std
        }
        torch.save(checkpoint, save_path)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

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
