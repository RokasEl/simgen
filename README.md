# EnergyMolecularDiffusion

The easiest way to setup the package is to install the associated model depository as shown below. This will automatically install this package and download the required model files. Make sure you're using `python>=3.10`!

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

The above will also install the `zndraw` package which allows the interactive use of the generative model. If you followed the above instructions, you can now try building a linker between two fragments by running:
```bash
zndraw ./data/zinc_fragments_difflinker.xyz
```

If you get an error regarding `libcublas` try running: ```pip uninstall nvidia_cublas_cu11```.

## Running on CSD3

Step 1:
Connect to CSD3 and note the exact login node you've connected to. E.g.
```bash
rokas@LAPTOP-G8ALNGOC:~$ ssh csd3
(base) [re344@login-q-1 ~]$
```
The login node is `login-q-1`.

Step 2:
Request an interactive GPU:
```bash
sintr -A ljc-re344-sl2-gpu -p ampere -N1 -n1 -t 1:0:0 --qos=INTR --gres=gpu:1
```
After `sintr -A` enter which account should be charged for you.

I find it convenient  to add an alias in my `~/.bashrc` to quickly request a gpu:

```bash
alias get_interactive_gpu="sintr -A ljc-re344-sl2-gpu -p ampere -N1 -n1 -t 1:0:0 --qos=INTR --gres=gpu:1"
```

Then you can request an interactive gpu simply by:

```bash
(base) [re344@login-q-1 data]$ get_interactive_gpu
sbatch: Loaded geopm plugin.
Waiting for JOBID 25203255 to start
(base) [re344@gpu-q-15 data]$
```
Note the exact gpu you've been allocated. In this case it is `gpu-q-15`.

Step 3:
Activate the appropriate python environment and launch zndraw on the cluster.

```bash
(base) [re344@gpu-q-15 data]$ conda activate test-env
(test-env) [re344@gpu-q-15 data]$ zndraw --no-browser --port 8081 ./zinc_fragments_difflinker.xyz
```
Replace `./zinc_fragments_difflinker.xyz` with an appropriate path for your use case. If you followed our install instructions, and are currently in the `MACE-models/` folder, `./data/zinc_fragments_difflinker.xyz` should work.
Take note of the port used in the command, in this case `8081`.

Step 4
On your local machine port forward the above port from the cluster to your machine. To do this run

```bash
 ssh -L 8081:gpu-q-15:8081 -fN re344@login-q-1.hpc.cam.ac.uk
```
Replace `gpu-q-15`, `login-q-1`, and `re344` with the resources you've been allocated and your crsid.

`zndraw` should now be available through your browser on `localhost:8081`.
