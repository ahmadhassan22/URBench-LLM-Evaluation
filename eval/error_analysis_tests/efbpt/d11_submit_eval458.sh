#!/bin/bash
# D11 RTG eval458 — the one held-out run (freeze §8; Amendment 2 A2.2-A2.5; docs/D11_GO_EVAL458.md).
# One SLURM job per student arm and per teacher reader; submission order = priority:
#   J0 regress (student, DEV200, TR rows 0-4, byte compare)
#   J1 teacher R1 eval458        afterok J0    03:00:00
#   J2 student A_r2_eval         afterok J0    08:00:00
#   J3 student TR                afterok J1    08:00:00
#   J4 student TR_wrong_eval     afterok J1    08:00:00
#   J5 teacher R3 eval458        afterok J0    03:00:00  --nice=100
#   J6 student TR_L1_eval        afterok J5    08:00:00  --nice=100
# No job computes or prints accuracy. The merge is run later, by hand, on CPU:
#   python eval/error_analysis_tests/efbpt/d11_student.py --set eval458 --mode FULL --eval-arm TR --merge
# A crashed arm is resubmitted with the same ARMS: it resumes from its last completed row (A2.5).
# Usage (from ~/URBench): bash eval/error_analysis_tests/efbpt/d11_submit_eval458.sh
set -euo pipefail
cd /mnt/home/user41/URBench
D=eval/error_analysis_tests/efbpt
jid() { grep -oE '[0-9]+$'; }

[ -f docs/D11_GO_EVAL458.md ] || { echo "docs/D11_GO_EVAL458.md missing: eval458 does not run (freeze §8)" >&2; exit 1; }
# eval458 runs once: refuse a second submission
if [ -e outputs/efbpt/d11/student/eval458 ] || [ -e outputs/efbpt/d11/teacher/eval458 ]; then
  echo "outputs/efbpt/d11/{student,teacher}/eval458 exists: eval458 was already started. Resubmit single crashed arms by hand (A2.5)." >&2; exit 1
fi
if squeue -h -u "$USER" -o '%j' | grep -q 'eval458'; then
  echo "eval458 jobs are already in the queue" >&2; exit 1
fi
# the jobs record git head / dirty: the D11 scripts and contracts must be committed
if [ -n "$(git status --porcelain -- $D docs)" ]; then
  echo "uncommitted changes under $D or docs: commit before submitting" >&2; exit 1
fi

J0=$(bash $D/d11_student.sbatch REGRESS | jid)
echo "J0 regress (student, DEV200)   $J0"
J1=$(READER=R1 SBATCH_DEP="$J0" bash $D/d11_teacher.sbatch FULL eval458 | jid)
echo "J1 teacher R1 eval458          $J1  (afterok:$J0)"
J2=$(ARMS=A_r2_eval EVAL_ARM=TR SBATCH_DEP="$J0" D11_TIME=08:00:00 bash $D/d11_student.sbatch FULL eval458 | jid)
echo "J2 student A_r2_eval           $J2  (afterok:$J0)"
J3=$(ARMS=TR EVAL_ARM=TR SBATCH_DEP="$J1" D11_TIME=08:00:00 bash $D/d11_student.sbatch FULL eval458 | jid)
echo "J3 student TR                  $J3  (afterok:$J1)"
J4=$(ARMS=TR_wrong_eval EVAL_ARM=TR SBATCH_DEP="$J1" D11_TIME=08:00:00 bash $D/d11_student.sbatch FULL eval458 | jid)
echo "J4 student TR_wrong_eval       $J4  (afterok:$J1)"
J5=$(READER=R3 SBATCH_DEP="$J0" D11_NICE=100 bash $D/d11_teacher.sbatch FULL eval458 | jid)
echo "J5 teacher R3 eval458          $J5  (afterok:$J0, nice 100)"
J6=$(ARMS=TR_L1_eval EVAL_ARM=TR SBATCH_DEP="$J5" D11_TIME=08:00:00 D11_NICE=100 bash $D/d11_student.sbatch FULL eval458 | jid)
echo "J6 student TR_L1_eval          $J6  (afterok:$J5, nice 100)"
echo "jobs: $J0 $J1 $J2 $J3 $J4 $J5 $J6"
