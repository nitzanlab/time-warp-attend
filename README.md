# Time-Warp-Attend

Dynamical systems across the sciences, from electrical circuits to ecological networks, undergo qualitative and often catastrophic changes in behavior, called
bifurcations, when their underlying parameters cross a threshold. Existing methods
predict oncoming catastrophes in individual systems but are primarily time-series-
based and struggle both to categorize qualitative dynamical regimes across diverse
systems and to generalize to real data. To address this challenge, we present here **time-warp-attend**, a
data-driven, physically-informed deep-learning framework for classifying dynamical regimes and characterizing bifurcation boundaries based on the extraction oftopologically invariant features. We focus on the paradigmatic case of the supercritical Hopf bifurcation, which is used to model periodic dynamics across a wide range of applications. 

![Time-warp-attend framework](https://github.com/nitzanlab/time-warp-attend/raw/main/.images/graphical_abs.png)

For more information see paper at [ICLR 2024](https://openreview.net/forum?id=Fj7Fzm5lWL).

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


## Generate classical,synthetic systems data
To view a few examples of the data generation process, see the notebook `notebooks/data.ipynb`.

To generate the data used in the paper, run the following commands:
```
# train datasets
twa generate-dataset --data-dir output/data/simple_oscillator_nsfcl  --train-size 12000 --test-size 1100 --data-name simple_oscillator  --augment-type NSF_CL
twa generate-dataset --data-dir output/data/simple_oscillator_noaug  --train-size 10000 --test-size 1000 --data-name simple_oscillator

# test datasets
twa generate-dataset --data-dir output/data/suphopf --test-size 1000 --data-name suphopf 
twa generate-dataset --data-dir output/data/lienard_poly --test-size 1000 --data-name lienard_poly 
twa generate-dataset --data-dir output/data/lienard_sigmoid --test-size 1000 --data-name lienard_sigmoid 
twa generate-dataset --data-dir output/data/vanderpol --test-size 1000 --data-name vanderpol 
twa generate-dataset --data-dir output/data/bzreaction --test-size 1000 --data-name bzreaction 
twa generate-dataset --data-dir output/data/selkov --test-size 1000 --data-name selkov 
```

## Generate repressilator data
In the notebook `notebooks/repressilator.ipynb`, we simulate the repressilator regulatory gene network for  cell trajectories varying in their transcription rate, $\alpha$, and the ratio of protein and mRNA degradation rates, $\beta$. From these, we generate respective vector fields across pTetR-pLacI phase space.

## Generate pancreas data
For the pancreas dataset, vector fields and their corresponding cell cycle score are generated in the notebook `notebooks/pancreas.ipynb`.

## Train models

To train a single model, see the notebook `notebooks/train.ipynb`.

To run multiple experiments, use the `twa train` command. For example, to train the models used in the paper, run the following commands:
```
twa train simple_oscillator_nsfcl output/
```

## Evaluate models

Statistical and visual evaluations of single runs are available in the notebook `notebooks/evaluate.ipynb`.

## Contact

Please get in touch [email](mailto:noa.moriel@mail.huji.ac.il).