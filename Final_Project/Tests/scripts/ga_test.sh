clear
set -e

instanceName=('tba1' 'tba2' 'tba3' 'tba4' 'tba5' 'tba6' 'tba7' 'tba8' 'tba9' 'tba10')
#Mu=('100' '200' '300' '400' '500' '600' '700' '800' '900' '1000')
#Lambda=('100' '200' '300' '400' '500' '600' '700' '800' '900' '1000')
#Phi=('0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1')
#Omega=('100' '200' '300' '400' '500' '600' '700' '800' '900' '1000')

#for value in "${Omega[@]}"; do
  for instance in "${instanceName[@]}"; do
    echo "$instance"
    python3 GeneticAlgorithm.py "$instance" "1000" "1000" "3" "0.25" "500" "11011011011" "summ_0008" &
    python3 GeneticAlgorithm.py "$instance" "1000" "1000" "3" "0.25" "500" "10101010101" "summ_0008" &
    python3 GeneticAlgorithm.py "$instance" "1000" "1000" "3" "0.25" "500" "11001100110" "summ_0008" &
    python3 GeneticAlgorithm.py "$instance" "1000" "1000" "3" "0.25" "500" "11111111011" "summ_0008" &
    python3 GeneticAlgorithm.py "$instance" "1000" "1000" "3" "0.25" "500" "11101110110" "summ_0008" &
    wait
    echo "All 5 executions completed!"
  done
#done
