#!/bin/bash
#SBATCH -w obl1
#SBATCH -p obl
#SBATCH -n1
#SBATCH -A prj_chess
#SBATCH --time=216:00:00
cd /home/ai164201/projects/smallm
source .venv/bin/activate
uv run --with jupyter jupyter lab --no-browser --port=8889 --ip 0.0.0.0
