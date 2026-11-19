#!/bin/bash
# Scale-matched null: the 100 fixed control directions per cell again, each
# stepped by +-2 within-class SD along its OWN direction (--control-unit own), so that the
# controls stay within the natural range of the texts as much as the class direction does.
# Output suffix _ownsd. Run on the login node from a clean shell:
#   ssh <cluster-login-host> 'bash -c "cd /work/<user>/latent-detection && bash slurm/steer_nulls_own_submit.sh"'
set -euo pipefail
X="--only-controls --controls pcrandom split --control-unit own --suffix _ownsd"
A=slurm/steer_nulls.sh; V=slurm/steer_nulls_v100.sh
sbatch $V journalism_seg gemma3-4b instruct raw 22 $X
sbatch $V journalism_seg gemma3-4b instruct raw 28 $X
sbatch $V chat_tw25 gemma3-4b instruct raw 17 $X
sbatch $V truth_opspam gemma3-4b base raw 23 --limit 300 $X
sbatch $V truth_opspam gemma3-4b instruct raw 23 --limit 300 $X
sbatch $V truth_neg_cities olmo2-7b base raw 21 --limit 300 --batch-size 16 $X
sbatch $V truth_neg_cities olmo2-7b sft raw 21 --limit 300 --batch-size 16 $X
for L in 24 30; do sbatch $A journalism_seg qwen3-8b instruct raw $L $X; sbatch $A chat_tw25 qwen3-8b instruct raw $L $X; done
sbatch $A chat_tw25 gemma3-12b instruct raw 31 $X
sbatch $A journalism_seg gemma3-12b instruct raw 31 $X
for v in base sft; do for L in 16 21 27; do sbatch $A textbook apertus15-8b $v raw $L $X; done; done
for v in base sft; do sbatch $A textbook olmo3-7b-think $v raw 21 --batch-size 8 $X; done
for v in base sft; do sbatch $A truth_neg_cities apertus15-8b $v raw 21 --limit 300 $X; done
for v in base instruct; do sbatch $A truth_opspam qwen3-8b $v raw 24 --limit 300 $X; done
squeue -u "$USER" -h -o "%j %T" | sort | uniq -c
