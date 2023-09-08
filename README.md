# Perceptual MAE Codebase

![poster_thumb_pwc](https://github.com/tractableai/perceptual-mae/assets/1108445/117ad9cf-c77d-41c9-b419-6ce2bcc03c1e)

This repository contains the source code to accompany the following paper:

```
@Article{PerceptualMAE2022,
  author  = {Samyakh Tukra and Frederick Hoffman and Ken Chatfield},
  journal = {arXiv:2212.14504},
  title   = {Improving Visual Representation Learning through Perceptual Understanding},
  year    = {2022},
}
```

### Fine-tuned Performance

We obtain the following fine-tuned results over ImageNet, setting state-of-the-art whilst being much more data and parameter efficient than alternate methods (for more details see the paper):

![recent_sota_wide](https://github.com/tractableai/perceptual-mae/assets/1108445/d1a21356-17d1-4a16-8ae2-3d0cd719db21)

## Setup

Optionally, set up a docker container to run the code, based on [NVIDIA CUDA images](https://hub.docker.com/r/nvidia/cuda):

```bash
cd docker
./build.sh
./start.sh
```

Following this, from within the container (or directly on your local system) set up
a Python environment using venv, and install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Preparing data

Place ImageNet tarballs in the `{data_dir}/public_datasets/imagenet` directory, where
`{data_dir}` is specified by the `user_config.data_dir` setting in the
`configs/user/sample.yaml` configuration file (by default `<current_user_homedir>/percmae_data/`):

```bash
mkdir /home/myuser/percmae_data/public_datasets/imagenet && cd "$_"
# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar  --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
```

## Training Process (base models)

### Starting a training run & model configs

In order to start a training run, you need to specify the model config to the `train.py` script. This can be done by providing:

1. The path to a model config file (using the `--model_config_path` argument)
2. (optional) The path to a model config variant file (using the `--model_variant_config_path` argument)

For example, for a masked image model, the following command can be used:

```bash
python3 train.py --model_config_path configs/models/masked_image.yaml
                 --model_variant_config_path configs/models/masked_image/msg_gan.yaml
                 --experiment_tag 230206
```

This will start a training run using ImageNet using the base `masked_image.yaml` config with the `msg_gan.yaml` variant.

The location of the model variants are in a subdirectory with the same name as the base model config:

| Model config                          | Variants directory                |
|---------------------------------------|-----------------------------------|
| `configs/models/masked_image.yaml`    | `configs/models/masked_image/`    |

By default, if a model variant is provided then an experiment name for the run will be automatically generated from the
name of the variant (in this case `msg_gan`). By specifying the `--experiment_tag` parameter as above, instead the format
of `<experiment_tag>_<variant_name>` will be used instead, allowing for multiple unique runs.

### Tracking experiments

The location of output data is based on values in the config file specified by the `--user_config_path` argument.
Normally there is no need to explicitly set this, and can use the default config at `configs/user/sample.yaml`.

This specifies an experiment directory will be created of the following format:

```
<current_user_homedir>/percmae_data/<experiment_name>
```

Tensorboard logs are stored in the `train_outputs/tensorboard_logs` subdirectory, and reconstructions generated during the
training process in the `mae_recon_out` subdirectory.

### Training parameters & Multi-GPU training

The training parameter config is specified by the `--default_config_path` arugment.
You can also customise the default config which will be used if no path is specified at `configs/default.yaml`,
which contains the parameters used for training runs in the paper. This also specifies the metrics to track
during training.

By default, all GPUs on the system are used for training (`trainer.params.devices` is set to -1).
To train on a specific GPU, set this to the 0-based index of the GPU to train on.

### Training times

The rough times to train 300 epochs over ImageNet (as in the paper) on 4xV100 GPUs is as follows:

| Method   | Time               |
|----------|--------------------|
| MAE      | 15 days (~2 weeks) |
| LS-GAN   | 35 days (~5 weeks) |
| MSG-GAN  | 65 days (~9 weeks) |

These times can be roughly halved when training on A100s and can be further boosted using distributed training.

## Fine-tuning for downstream tasks

Use the `masked_image_downstream_classifier` model type:

```bash
python3 train.py --model_config_path configs/models/masked_image_downstream_classifier.yaml
```

## Additional references

* [Blog post](https://tractable.ai/en/resources/efficient-learning-of-domain-specific-visual-cues-with-self-supervision)
* [Video introduction](https://youtu.be/RMc9mK7n8qk)
