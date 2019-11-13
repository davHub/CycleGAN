# CycleGAN

A quick implementation of [CycleGAN](https://junyanz.github.io/CycleGAN/) using tensorflow 1.
The implementation is similar to the original paper.

### CycleGAN train/test
- Download repository and install dependancies
```bash
git clone https://github.com/blavad/CycleGAN.git
pip intall -e CycleGAN
cd CycleGAN
```

- Train a model from scratch (with standard settings)
```bash
python main.py -dA /path/to/datasetA -dB /path/to/datasetB
```

- Train a model from pretrain model (with standard settings)
```bash
python main.py -dA /path/to/datasetA -dB /path/to/datasetB --restore path/to/model
```

- Test a model
```bash
python main.py -dA /path/to/datasetA -dB /path/to/datasetB --restore path/to/model --testing
```
