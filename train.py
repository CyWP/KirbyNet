import time
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from exceptions import MaxPooledException
from test import run_test

if __name__ == "__main__":
    opt = TrainOptions().parse()
    dataset = DataLoader(opt)
    dataset_size = len(dataset)
    print("#training meshes = %d" % dataset_size)

    model = create_model(opt)
    writer = Writer(opt)
    total_steps = 0
    total_steps = 0
    model.set_mean_std(dataset.mean, dataset.std)

    def train_batch(i, data, retry, model, epoch, missed_batches):
        iter_data_time = time.time()
        try:
            model.set_input(data)
            if epoch == opt.epoch_count:
                writer.set_init_energy(model.get_mesh_energy())
        except Exception as e:
            if retry:
                missed_batches.append((i, data))
            print(str(e))
            print("Skipping batch")
            return False
        try:
            model.optimize_parameters()
        except Exception as e:
            if retry:
                missed_batches.append((i, data))
            print(str(e))
            print("Skipping batch")
            return False

        return True
    
    try:
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            failed_batches = 0
            successful_batches = 0
            missed_batches = []
            print(f"---------\nEpoch {epoch}\n----------\n")
            for i, data in enumerate(dataset):
                if train_batch(i, data, True, model, epoch, missed_batches):
                    successful_batches += 1
                    writer.set_epoch_stats(epoch, model.get_training_stats())
                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    if total_steps % opt.print_freq == 0:
                        loss = model.loss
                        t = (time.time() - iter_start_time) / opt.batch_size
                        writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                        writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
                        
                    if i % opt.save_latest_freq == 0:
                        print(
                            "saving the latest model (epoch %d, total_steps %d)"
                            % (epoch, total_steps)
                        )
                        model.save_network("latest")

            for pair in missed_batches:
                i, data = pair
                if train_batch(i, data, False, model, epoch, missed_batches):
                    writer.set_epoch_stats(epoch, model.get_training_stats())
                    iter_start_time = time.time()
                    if total_steps % opt.print_freq == 0:
                        t_data = iter_start_time - iter_data_time
                    total_steps += opt.batch_size
                    epoch_iter += opt.batch_size

                    if total_steps % opt.print_freq == 0:
                        loss = model.loss
                        t = (time.time() - iter_start_time) / opt.batch_size
                        writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                        writer.plot_loss(loss, epoch, epoch_iter, dataset_size)
                        
                    if i % opt.save_latest_freq == 0:
                        print(
                            "saving the latest model (epoch %d, total_steps %d)"
                            % (epoch, total_steps)
                        )
                        model.save_network("latest")
                    successful_batches += 1
                else:
                    failed_batches += 1
            if epoch % opt.save_epoch_freq == 0:
                print(
                    "saving the model at the end of epoch %d, iters %d"
                    % (epoch, total_steps)
                )
                model.save_network("latest")
                model.save_network(epoch)

            print(
                "End of epoch %d / %d \t Time Taken: %d sec"
                % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
            )
            writer.print_epoch_stats(epoch)
            writer.set_failure_rate(epoch, successful_batches, failed_batches)
            model.update_learning_rate()
            if opt.verbose_plot:
                writer.plot_model_wts(model, epoch)
    finally:
        print("Saving training statistics...")
        writer.dump_training_stats()
        print("Done.")
        writer.close()
