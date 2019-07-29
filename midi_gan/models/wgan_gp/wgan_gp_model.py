import os

import numpy as np
from keras.utils import plot_model

from midi_gan.models.abstract_gan.abstract_gan_model import AbstractGAN
from midi_gan.models.wgan_gp import wgan_gp_utils
from midi_gan.utils.utils import plot_save_losses, save_samples

generator_lr = 1e-4
critic_lr = 1e-4
gradient_penalty_weight = 10
batch_size = 64
n_generator = 1
n_critic = 5


class WGAN_GP(AbstractGAN):
    def _save_latent_space(self):
        pass

    def _generate_dataset(self):
        pass

    def __init__(self, run_dir, outputs_dir, model_dir, generated_datasets_dir, epochs, output_save_frequency,
                 model_save_frequency, loss_save_frequency, latent_space_save_frequency, dataset_generation_frequency,
                 dataset_size, sequence_length, latent_dim, rate, playback_nsamps):

        super(WGAN_GP, self).__init__(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                                      generated_datasets_dir=generated_datasets_dir,
                                      epochs=epochs, output_save_frequency=output_save_frequency,
                                      model_save_frequency=model_save_frequency,
                                      loss_save_frequency=loss_save_frequency,
                                      latent_space_save_frequency=latent_space_save_frequency,
                                      dataset_generation_frequency=dataset_generation_frequency,
                                      dataset_size=dataset_size,
                                      sequence_length=sequence_length,
                                      latent_dim=latent_dim)
        self._generator_lr = generator_lr
        self._critic_lr = critic_lr
        self._gradient_penalty_weight = gradient_penalty_weight
        self._batch_size = batch_size
        self._n_generator = n_generator
        self._n_critic = n_critic
        self._rate = rate
        self._playback_nsamps = playback_nsamps

        self._losses = [[], []]

        self._build_models()
        self._save_models_architectures()

    def _build_models(self):
        self._generator = wgan_gp_utils.build_generator(self._latent_dim, self._sequence_length)
        self._critic = wgan_gp_utils.build_critic(self._sequence_length)

        self._generator_model = wgan_gp_utils.build_generator_model(self._generator,
                                                                    self._critic,
                                                                    self._latent_dim,
                                                                    self._generator_lr)

        self._critic_model = wgan_gp_utils.build_critic_model(self._generator,
                                                              self._critic,
                                                              self._latent_dim,
                                                              self._sequence_length,
                                                              self._batch_size,
                                                              self._critic_lr,
                                                              self._gradient_penalty_weight)

    def _save_models_architectures(self):
        plot_model(self._generator, to_file=self._run_dir + '/generator.png')
        plot_model(self._critic, to_file=self._run_dir + '/critic.png')

    def train(self, dataset):
        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(self._n_critic):
                indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
                real_samples = dataset[indexes]
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [real_samples, noise]

                critic_losses.append(self._critic_model.train_on_batch(inputs, [ones, neg_ones, zeros])[0])
            critic_loss = np.mean(critic_losses)

            generator_losses = []
            for _ in range(self._n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))

                generator_losses = self._generator_model.train_on_batch(noise, ones)
            generator_loss = np.mean(generator_losses)

            generator_loss = float(-generator_loss)
            critic_loss = float(-critic_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(critic_loss)

            print("%d [C loss: %+.6f] [G loss: %+.6f]" % (self._epoch, critic_loss, generator_loss))

            if self._epoch % self._loss_save_frequency == 0 and self._loss_save_frequency > 0:
                self._save_losses()

            if self._epoch % self._output_save_frequency == 0 and self._output_save_frequency > 0:
                self._save_outputs()

            # if self._epoch % self._latent_space_save_frequency == 0 and self._latent_space_save_frequency > 0:
            #     self._save_latent_space()
            #
            if self._epoch % self._model_save_frequency == 0 and self._model_save_frequency > 0:
                self._save_models()

            # if self._epoch % self._dataset_generation_frequency == 0 and self._dataset_generation_frequency > 0:
            #     self._generate_dataset()

        self._generate_dataset()
        self._save_losses()
        self._save_models()
        self._save_outputs()
        self._save_latent_space()

        return self._losses

    def _save_outputs(self):
        noise = np.random.normal(0, 1, (10, self._latent_dim))
        generated_samples = self._generator.predict(noise)

        generated_samples[:, :, :, 0] *= 108.0
        generated_samples[:, :, :, 1] *= 15.0
        generated_samples[:, :, :, 2] *= 3.0

        generated_samples = generated_samples.astype(np.int)

        generated_samples[:, :, 3, 0] = np.clip(generated_samples[:, :, 3, 0], 0, 16)
        generated_samples[:, :, 3, 2] = np.clip(generated_samples[:, :, 3, 2], 0, 1)
        generated_samples[:, :, 2, 1:] *= 0

        save_samples(generated_samples, self._rate, self._playback_nsamps, self._outputs_dir, self._epoch)

    def _save_losses(self):
        plot_save_losses(self._losses[:2], ['generator', 'critic'], self._outputs_dir, 'gan_loss')

    def _save_models(self):
        root_dir = self._model_dir + str(self._epoch) + '/'
        os.mkdir(root_dir)
        self._critic_model.save(root_dir + 'critic_model.h5')
        self._generator_model.save(root_dir + 'generator.h5')
        self._generator.save(root_dir + 'generator.h5')
        self._critic.save(root_dir + 'critic.h5')
