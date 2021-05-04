#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --job-name=jupyter-notebook

# get tunneling info
XDG_RUNTIME_DIR=""


node=$(hostname -s  )
user=$(whoami)
port=38689


# print tunneling instructions jupyter-log
echo -e "
Command to create ssh tunnel:
ssh -L ${port}:${node}:${port} ${user}@curnagl.dcsr.unil.ch

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"




# load environment, e.g. set virtualenv, environment variables, etc


source /scratch/dmoi/miniconda/etc/profile.d/conda.sh
conda activate ML

# Run Jupyter

jupyter-lab --no-browser --port=${port} --ip=${node}
