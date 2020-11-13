#!/usr/bin/env python

from distutils.core import setup

setup(name='CycleGAN',
    version='1.0',
    description='Cycle GAN tools',
    author='David ALBERT',
    author_email='david.albert@insa-rouen.fr',
    install_requires=['tensorflow==2.3.1', 'numpy', 'opencv-python', 'matplotlib'],
)