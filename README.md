# Time-Warp-Attend



## How to install time-warp-attend
Clone this repository.

Create a clean environment with `conda` using the `environment.yml` file from this repository:

```
conda env create -f environment.yml
```


Activate the environment and install the package from parent directory:

```
conda activate twa
pip install -e time-warp-attend
```

## Temporary tutorial

### Generate synthetic systems data
To view a few examples of the data generation process, see the notebook `notebooks/data.ipynb`.

To generate the data used in the paper, run the following commands:
```
# train datasets
twa generate-dataset --data-dir output/data/simple_oscillator_nsfcl  --train-size 12000 --test-size 1100 --data-name simple_oscillator  --augment-type NSF_CL --device cuda
twa generate-dataset --data-dir output/data/simple_oscillator_noaug  --train-size 10000 --test-size 1000 --data-name simple_oscillator  --device cuda

# test datasets
twa generate-dataset --data-dir output/data/selkov  --test-size 1000 --data-name selkov  --device cuda
twa generate-dataset --data-dir output/data/bzreaction  --test-size 1000 --data-name bzreaction  --device cuda
twa generate-dataset --data-dir output/data/lienard_poly  --test-size 1000 --data-name lienard_poly  --device cuda
twa generate-dataset --data-dir output/data/lienard_sigmoid  --test-size 1000 --data-name lienard_sigmoid  --device cuda
```
### Generate pancreas data
For the pancreas dataset, vector fields and their corresponding cell cycle score are generated in the notebook `notebooks/pancreas.ipynb`.

### Train models

To train a single model, see the notebook `notebooks/train.ipynb`.

To run multiple experiments, use the `twa train` command. For example, to train the models used in the paper, run the following commands:
```
twa train simple_oscillator_nsfcl output/
```

### Evaluate models

Statistical and visual evaluations of single runs are available in the notebook `notebooks/evaluate.ipynb`.