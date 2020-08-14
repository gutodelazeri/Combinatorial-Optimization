clear
set -e

instanceName=('tba1' 'tba2' 'tba3' 'tba4' 'tba5' 'tba6' 'tba7' 'tba8' 'tba9' 'tba10')
Mu=('100' '200' '300' '400' '500' '600' '700' '800' '900' '1000')

for value in "${Mu[@]}"; do
  for instance in "${instanceName[@]}"; do
    echo "$value - $instance"
    python3 GeneticAlgorithm.py "$instance" "$value" "$value" "10" "0.5" "500" "11011011011" "summ_0003"
    python3 GeneticAlgorithm.py "$instance" "$value" "$value" "10" "0.5" "500" "10101010101" "summ_0003"
    python3 GeneticAlgorithm.py "$instance" "$value" "$value" "10" "0.5" "500" "11001100110" "summ_0003"
  done
done