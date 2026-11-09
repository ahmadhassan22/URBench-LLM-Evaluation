#!/bin/bash
# D11 RTG chain (docs/EFBPT_D11_RTG_FREEZE.md §9; Amendment 1 A1.3):
#   TEST teacher -> TEST student -> FULL teacher -> FULL student, each --dependency=afterok.
# Usage (from ~/URBench): bash eval/error_analysis_tests/efbpt/d11_submit.sh [dev200]
set -euo pipefail
cd /mnt/home/user41/URBench
SET="${1:-dev200}"
D=eval/error_analysis_tests/efbpt
jid() { grep -oE '[0-9]+$'; }
T_TEST=$(bash $D/d11_teacher.sbatch TEST "$SET" | jid)
echo "TEST teacher  $T_TEST"
S_TEST=$(SBATCH_DEP="$T_TEST" bash $D/d11_student.sbatch TEST "$SET" | jid)
echo "TEST student  $S_TEST  (afterok:$T_TEST)"
T_FULL=$(SBATCH_DEP="$S_TEST" bash $D/d11_teacher.sbatch FULL "$SET" | jid)
echo "FULL teacher  $T_FULL  (afterok:$S_TEST)"
S_FULL=$(SBATCH_DEP="$T_FULL" bash $D/d11_student.sbatch FULL "$SET" | jid)
echo "FULL student  $S_FULL  (afterok:$T_FULL)"
echo "jobs: $T_TEST $S_TEST $T_FULL $S_FULL"
