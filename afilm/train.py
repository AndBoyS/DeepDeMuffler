from pathlib import Path

import click
from afilm.models.afilm import get_afilm
from afilm.models import get_tfilm
from .utils import load_audio_dataset
import tensorflow as tf
from tensorflow import keras


tf.compat.v1.disable_eager_execution()


class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(CustomCheckpoint, self).__init__()
        self.file_path = file_path
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.file_path)


@click.command()
@click.option('--model', default='afilm', type=click.Choice(['afilm', 'tfilm']), help='model to train')
@click.option('--data_dir', required=True)
@click.option('--epochs', type=int, default=20, help='number of epochs to train')
@click.option('--batch_size', type=int, default=16, help='training batch size')
@click.option('--logname', default='tmp-run', help='folder where logs will be stored')
@click.option('--layers', default=4, type=int, help='number of layers in each of the D and U halves of the network')
@click.option('--lr', default=3e-4, type=float, help='learning rate')
@click.option('--save_path', default="model.h5", help='path to save the model')
@click.option('--r', type=int, default=4, help='upscaling factor')
@click.option('--pool_size', type=int, default=4, help='size of pooling window')
@click.option('--strides', type=int, default=4, help='pooling stide')
def train(**kwargs):
    data_dir = Path(kwargs['data_dir'])
    raw_audio_dir = data_dir / 'raw'
    muffled_audio_dir = data_dir / 'muffled'
    x_train, y_train = load_audio_dataset(raw_audio_dir, muffled_audio_dir)

    model = get_model(kwargs['model'], kwargs['layers'], kwargs['r'])
    opt = keras.optimizers.Adam(learning_rate=kwargs['lr'])
    model_checkpoint_callback = CustomCheckpoint(file_path=kwargs['save_path'])
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
    model.compile(optimizer=opt, loss='mse', metrics=metrics)
    model.fit(x_train, y_train,
              batch_size=kwargs['batch_size'],
              epochs=kwargs['epochs'],
              callbacks=[model_checkpoint_callback])


name_to_model = {
    'tfilm': get_tfilm,
    'afilm': get_afilm,
}


def get_model(model_name, layers, r):
    assert model_name in name_to_model, 'Invalid model'
    model = name_to_model[model_name](n_layers=layers, scale=r)
    return model


def main():
    train(prog_name='train')


if __name__ == '__main__':
    main()
