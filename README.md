# MIDI-GAN
[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![HitCount](http://hits.dwyl.io/HitLuca/MIDI-GAN.svg)](http://hits.dwyl.io/HitLuca/MIDI-GAN)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HitLuca/MIDI-GAN.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/MIDI-GAN/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HitLuca/MIDI-GAN.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/MIDI-GAN/context:python)

## Description
Using GANs to generate MIDI music based on popular video-games. The input comes from the [NES music database](https://github.com/chrisdonahue/nesmdb).

### Models
This project uses a personal implementation of WGAN-GP, employing residual connections and 1-dimensional convolutions to generate music.

### Project structure
The GAN model is located in the [models](midi_gan/models) folder, the input dataset is downloaded and stored in the [data](data) folder. The main function is [train_model.py](midi_gan/train_model.py).

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)
