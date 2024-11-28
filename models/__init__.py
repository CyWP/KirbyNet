def create_model(opt):
    if opt.dataset_mode == "unsupervised":

        from .mesh_smoothing import SmoothingModel  # todo - get rid of this ?
        model = SmoothingModel(opt)

    else:

        from .mesh_classifier import ClassifierModel
        model = ClassifierModel(opt)

    return model
