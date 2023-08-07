# EnergyMolecularDiffusion

The easiest way to setup the package is to install the associated model depository as shown below. This will automatically install this package and download the required model files.

```bash
git clone git@github.com:RokasEl/MACE-Models.git
pip install ./MACE-Models
cd ./MACE-Models ; dvc pull
moldiff init .
```
The above will also install the `zndraw` package which allows the interactive use of the generative model. Run `zndraw YOUR_XYZ_FILE` to give it a try!
