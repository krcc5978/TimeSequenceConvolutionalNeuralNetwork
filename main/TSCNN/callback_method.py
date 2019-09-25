from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard


def log(path):
    return TensorBoard(log_dir=path)


def model_checkpoint(path, monitor, save_weights_only, save_best_only, period):
    return ModelCheckpoint(path, monitor=monitor, save_weights_only=save_weights_only, save_best_only=save_best_only,
                           period=period)


def reduce_lron_plateau(monitor, factor, patience, verbose):
    return ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=verbose)