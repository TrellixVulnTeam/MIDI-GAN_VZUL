from functools import partial

from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Activation, Conv1D, MaxPooling1D, \
    UpSampling1D, GlobalAveragePooling1D, Add
from keras.optimizers import Adam

from midi_gan.utils.gan_utils import RandomWeightedAverage, gradient_penalty_loss, set_model_trainable, wasserstein_loss


def build_generator(latent_dim, sequence_length, kernel_size=3, filters=8):
    image_size = 16
    filters *= int(sequence_length / image_size / 2)

    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size * 8)(generated)
    generated = Activation('relu')(generated)

    generated = Reshape((image_size, 8))(generated)
    generated = Conv1D(filters, kernel_size, padding='same')(generated)

    while image_size != sequence_length:
        generated = UpSampling1D()(generated)

        shortcut = generated
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

        generated = Activation('relu')(generated)
        generated = Conv1D(filters, kernel_size, padding='same')(generated)

        generated = Activation('relu')(generated)
        generated = Conv1D(filters, kernel_size, padding='same')(generated)

        generated = Add()([shortcut, generated])
        filters = int(filters / 2)
        image_size *= 2

    generated = Activation('relu')(generated)
    generated = Conv1D(4 * 3, kernel_size, activation='sigmoid', padding='same')(generated)
    generated = Reshape((sequence_length, 4, 3))(generated)

    generator = Model(generator_inputs, generated, name='generator')
    print(generator.summary())
    return generator


def build_critic(sequence_length, kernel_size=3, filters=8):
    critic_inputs = Input((sequence_length, 4, 3))
    criticized = critic_inputs

    criticized = Reshape((sequence_length, 4 * 3))(criticized)
    criticized = Conv1D(filters, 1, padding='same')(criticized)

    while sequence_length != 16:
        shortcut = criticized
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

        criticized = LeakyReLU(0.2)(criticized)
        criticized = Conv1D(filters, kernel_size, padding='same')(criticized)

        criticized = LeakyReLU(0.2)(criticized)
        criticized = Conv1D(filters, kernel_size, padding='same')(criticized)

        criticized = Add()([shortcut, criticized])
        criticized = MaxPooling1D()(criticized)

        filters *= 2
        sequence_length = int(sequence_length / 2)

    criticized = LeakyReLU(0.2)(criticized)
    criticized = GlobalAveragePooling1D()(criticized)
    criticized = Dense(1)(criticized)

    critic = Model(critic_inputs, criticized, name='critic')
    print(critic.summary())
    return critic


def build_generator_model(generator, critic, latent_dim, generator_lr):
    set_model_trainable(generator, True)
    set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_criticized = critic(generated_samples)

    generator_model = Model(noise_samples, generated_criticized, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0, beta_2=0.9),
                            loss=wasserstein_loss)
    return generator_model


def build_critic_model(generator, critic, latent_dim, sequence_length, batch_size, critic_lr, gradient_penalty_weight):
    set_model_trainable(generator, False)
    set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((sequence_length, 4, 3))

    generated_samples = generator(noise_samples)
    generated_criticized = critic(generated_samples)
    real_criticized = critic(real_samples)

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic(averaged_samples)

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples],
                         [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0, beta_2=0.9),
                         loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    return critic_model
