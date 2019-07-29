import abc


class AbstractGAN:
    __metaclass__ = abc.ABCMeta

    def __init__(self, run_dir, outputs_dir, model_dir, generated_datasets_dir, epochs, output_save_frequency,
                 model_save_frequency, loss_save_frequency,
                 latent_space_save_frequency, dataset_generation_frequency, dataset_size,
                 sequence_length, latent_dim):
        self._run_dir = run_dir
        self._outputs_dir = outputs_dir
        self._model_dir = model_dir
        self._generated_datasets_dir = generated_datasets_dir

        self._sequence_length = sequence_length
        self._epochs = epochs
        self._output_save_frequency = output_save_frequency
        self._model_save_frequency = model_save_frequency
        self._loss_save_frequency = loss_save_frequency
        self._latent_space_save_frequency = latent_space_save_frequency
        self._latent_dim = latent_dim

        self._dataset_generation_frequency = dataset_generation_frequency
        self._dataset_size = dataset_size

        self._epoch = 0

    @abc.abstractmethod
    def _build_models(self):
        pass

    @abc.abstractmethod
    def train(self, dataset):
        pass

    @abc.abstractmethod
    def _save_models_architectures(self):
        pass

    @abc.abstractmethod
    def _save_outputs(self):
        pass

    @abc.abstractmethod
    def _save_latent_space(self):
        pass

    @abc.abstractmethod
    def _save_losses(self):
        pass

    @abc.abstractmethod
    def _save_models(self):
        pass

    @abc.abstractmethod
    def _generate_dataset(self):
        pass
