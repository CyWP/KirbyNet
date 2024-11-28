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
    err_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            try:
                model.set_input(data)
            except Exception as e:
                print(f"Failed to set input for this batch, will be skipped.")
                err_steps += opt.batch_size
                continue
            try:
                model.optimize_parameters()
            except Exception as e:
                if isinstance(e, MaxPooledException):
                    print(str(e))
                    print("Skipping training for this batch.")
                    err_steps += opt.batch_size
                    continue
                else:
                    raise(e)
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

            iter_data_time = time.time()
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
        print(f"Total successful steps: {total_steps}\nFailed steps: {err_steps} ({100*err_steps/(err_steps+total_steps)}%)")
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

    writer.close()
