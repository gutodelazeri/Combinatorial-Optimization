clear
set -e

Mu=('100' '200' '300' '400' '500' '600' '700' '800' '1000' '1100' '1200' '1300' '1400' '1500')
Lambda=('10' '20' '30' '40' '50' '60' '70' '80' '90' '100' '110' '120' '130' '140' '150' '160' '170' '180' '190' '200')
k=('3' '4' '5' '6' '7' '7' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20')
Phi=('0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1')
Omega=('100' '200' '300' '400' '500' '600' '700' '800' '900' '1000')

for value in "${Mu[@]}"; do
    clear
    echo "> Mu = $value"
    python3 GeneticAlgorithm.py "tba1" "$value" "100" "10" "0.5" "500" "11011011011" "Mu"
    python3 GeneticAlgorithm.py "tba2" "$value" "100" "10" "0.5" "500" "10101010101" "Mu"
    python3 GeneticAlgorithm.py "tba3" "$value" "100" "10" "0.5" "500" "11001100110" "Mu"
done

for value in "${Lambda[@]}"; do
    clear
    echo "> Lambda = $value"
    python3 GeneticAlgorithm.py "tba1" "700" "$value" "10" "0.5" "500" "11011011011" "Lambda"
    python3 GeneticAlgorithm.py "tba2" "700" "$value" "10" "0.5" "500" "10101010101" "Lambda"
    python3 GeneticAlgorithm.py "tba3" "700" "$value" "10" "0.5" "500" "11001100110" "Lambda"
done

for value in "${k[@]}"; do
  clear
  echo "> k = $value"
  python3 GeneticAlgorithm.py "tba1" "700" "100" "$value" "0.5" "500" "11011011011" "k"
  python3 GeneticAlgorithm.py "tba2" "700" "100" "$value" "0.5" "500" "10101010101" "k"
  python3 GeneticAlgorithm.py "tba3" "700" "100" "$value" "0.5" "500" "11001100110" "k"
done

for value in "${Phi[@]}"; do
  clear
  echo "> Phi = $value"
  python3 GeneticAlgorithm.py "tba1" "700" "100" "10" "$value" "500" "11011011011" "Phi"
  python3 GeneticAlgorithm.py "tba2" "700" "100" "10" "$value" "500" "10101010101" "Phi"
  python3 GeneticAlgorithm.py "tba3" "700" "100" "10" "$value" "500" "11001100110" "Phi"
done

for value in "${Omega[@]}"; do
  clear
  echo "> Omega = $value"
  python3 GeneticAlgorithm.py "tba1" "700" "100" "10" "0.5" "$value" "11011011011" "Omega"
  python3 GeneticAlgorithm.py "tba2" "700" "100" "10" "0.5" "$value" "10101010101" "Omega"
  python3 GeneticAlgorithm.py "tba3" "700" "100" "10" "0.5" "$value" "11001100110" "Omega"
done
