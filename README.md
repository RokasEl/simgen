# EnergyMolecularDiffusion

To install

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
git clone -b extract-embeddings git@github.com:RokasEl/mace.git
pip install ./mace
pip install git+https://github.com/RokasEl/hydromace
pip install zndraw==0.2.0a4
git clone git@github.com:RokasEl/MACE-Models.git
pip install ./MACE-Models
cd ./MACE-Models ; dvc pull
git clone git@github.com:RokasEl/EnergyMolecularDiffusion.git
pip install ./EnergyMolecularDiffusion
```

To setup `zndraw` features run:

```bash
cd ./EnergyMolecularDiffusion
python ./moldiff_zndraw/setup_defaults.py --default_model_path PATH_TO_MACE-MODELS_CLONE
```
Now when you launch `zndraw` the generative model features should load automatically.
