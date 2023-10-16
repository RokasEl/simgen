# EnergyMolecularDiffusion

The easiest way to setup the package is to install the associated model depository as shown below. This will automatically install this package and download the required model files. Make sure you're using `python>=3.10`!

```bash
git clone git@github.com:RokasEl/MACE-Models.git
pip install ./MACE-Models
cd ./MACE-Models ; dvc pull
pip install git+ssh://git@github.com/RokasEl/EnergyMolecularDiffusion.git
moldiff_init .
```

If you use GPG keys for authentication, instead use:
```bash
pip install git+https://github.com/RokasEl/EnergyMolecularDiffusion.git
```

The above will also install the `zndraw` package which allows the interactive use of the generative model. If you followed the above instructions, you can now try building a linker between two fragments by running:
```bash
moldiff_server --device cuda &
zndraw ./data/zinc_fragments_difflinker.xyz
```

If you get an error regarding `libcublas` try running: ```pip uninstall nvidia_cublas_cu11```.
