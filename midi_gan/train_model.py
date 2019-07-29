from midi_gan.models.wgan_gp.wgan_gp_model import WGAN_GP
from midi_gan.utils import utils

DATASET_FILEPATH = '../data/nesmdb24_exprsco/train.npy'
OUTPUTS_DIR = '../outputs/'
EPOCHS = 10000
LOSS_SAVE_FREQUENCY = 25
OUTPUT_SAVE_FREQUENCY = 125
LATENT_SPACE_SAVE_FREQUENCY = 10
MODEL_SAVE_FREQUENCY = -1
DATASET_GENERATION_FREQUENCY = -1
DATASET_SIZE = 1000

LATENT_DIM = 10
SEQUENCE_LENGTH = 256
PLAYBACK_NSAMPS = 645.825561905 * SEQUENCE_LENGTH
RATE = 24.0


def train():
    # FIRST TIME, CALL THE create_datasets() function to download and store the dataset

    dataset = utils.load_midi_dataset(DATASET_FILEPATH, SEQUENCE_LENGTH)

    run_dir, outputs_dir, model_dir, generated_datasets_dir = utils.generate_run_dir(OUTPUTS_DIR, 'wgan_gp')

    model = WGAN_GP(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                    generated_datasets_dir=generated_datasets_dir, epochs=EPOCHS,
                    output_save_frequency=OUTPUT_SAVE_FREQUENCY, model_save_frequency=MODEL_SAVE_FREQUENCY,
                    loss_save_frequency=LOSS_SAVE_FREQUENCY,
                    latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                    dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                    dataset_size=DATASET_SIZE, sequence_length=SEQUENCE_LENGTH, latent_dim=LATENT_DIM,
                    playback_nsamps=PLAYBACK_NSAMPS, rate=RATE)

    losses = model.train(dataset)
    return losses


if __name__ == "__main__":
    train()
