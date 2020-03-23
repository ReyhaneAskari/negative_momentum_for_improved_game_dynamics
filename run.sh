#!/usr/bin/env bash

python gan_cifar10.py 'mixed_adam_plus' 0.00003 -0.5 'no' 1
python gan_cifar10.py 'mixed_adam_plus' 0.00003 -0.5 'yes' 1
python gan_cifar10.py 'mixed_adam_plus' 0.00001 -0.5 'no' 1
python gan_cifar10.py 'mixed_adam_plus' 0.00001 -0.5 'yes' 1
python gan_cifar10.py 'mixed_adam_plus' 0.0001 -0.5 'no' 1
