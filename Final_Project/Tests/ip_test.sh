clear
set -e

instanceName=('tba1' 'tba2' 'tba3' 'tba4' 'tba5' 'tba6' 'tba7' 'tba8' 'tba9' 'tba10')

for instance in "${instanceName[@]}"; do
  echo "$instance"
  python3 IPSolver.py "$instance"
done