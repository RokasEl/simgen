# EnergyMolecularDiffusion

The easiest way to setup the package is to install the associated model depository as shown below. This will automatically install this package and download the required model files.

```bash
git clone git@github.com:RokasEl/MACE-Models.git
pip install ./MACE-Models
cd ./MACE-Models ; dvc pull
pip install git+ssh://git@github.com/RokasEl/EnergyMolecularDiffusion.git@load_models_from_remote
moldiff init .
```

If you use GPG keys for authentication, instead use:
```bash
pip install git+https://github.com/RokasEl/EnergyMolecularDiffusion.git@load_models_from_remote
```

The above will also install the `zndraw` package which allows the interactive use of the generative model. Run `zndraw YOUR_XYZ_FILE` to give it a try!
