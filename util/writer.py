import os
import time
import numpy as np
import pickle as pkl

try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print("tensorboard X not installed, visualizing wont be available")
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.stats_file = os.path.join(self.save_dir, "train_stats.pkl")
        self.log_name = os.path.join(self.save_dir, "log.txt")
        self.stats = {}
        self.init_energy = np.empty((0,), dtype=float)
        #
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(comment=opt.name)
        else:
            self.display = None

    def print_epoch_stats(self, epoch):
        epoch = str(epoch)
        stats = self.stats[epoch]
        mean = np.mean(stats['mean'])
        sqmean = np.mean(np.array(stats['mean'])**2)
        var = np.mean(stats['var'])
        norm = np.mean(stats['norm'])
        energy = np.mean(stats['energy'])
        init_e = np.mean(self.init_energy)
        print(f"\nMean, Squared mean, variance of output features: {mean}, {sqmean}, {var}\nMean of the norm of batch outputs: {norm}\nInitial mean curvature energy of meshes: {init_e}, Output mean curvature energy: {energy}\n")
        if 'fail' in stats.keys():
            print(f"Failed batches: {self.fail*100}%.")

    def set_init_energy(self, energy):
        self.init_energy = np.concatenate((self.init_energy, energy))

    def set_failure_rate(self, epoch, s, f):
        self.stats[str(epoch)]['fail'] = s/(s+f)

    def set_epoch_stats(self, epoch, batch_stats):
        epoch = str(epoch)
        if epoch not in self.stats.keys():
            self.stats[epoch] = {}
            self.stats[epoch]['norm'] = []
            self.stats[epoch]['mean'] = []
            self.stats[epoch]['var'] = []
            self.stats[epoch]['energy'] = np.empty((0,), dtype=float)
        self.stats[epoch]['norm'].append(batch_stats['norm'])
        self.stats[epoch]['mean'].append(batch_stats['mean'])
        self.stats[epoch]['var'].append(batch_stats['var'])
        self.stats[epoch]['energy'] = np.concatenate((self.stats[epoch]['energy'], batch_stats['energy']))

    def dump_training_stats(self):
        with open(self.stats_file, 'wb') as f:
            pkl.dump(self.stats, f)

    def print_current_losses(self, epoch, i, losses, t, t_data):
        """prints train loss to terminal / file"""
        message = "(epoch: %d, iters: %d, time: %.3f, data: %.3f) loss: %.3f " % (
            epoch,
            i,
            t,
            t_data,
            losses.item(),
        )
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

    def plot_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display:
            self.display.add_scalar("data/train_loss", loss, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch
                )

    def close(self):
        if self.display is not None:
            self.display.close()
