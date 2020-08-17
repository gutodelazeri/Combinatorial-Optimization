# Usage

Inside the directory named src:  
> $ python3 main.py instanceName method \[-h] \[-f OUTPUTFILE] \[-v] \[-t TIMELIMIT] \[-m MU] \[-l LAMBDA] \[-p PHI] \[-o OMEGA]
           
* instanceName: Name of the instance WITHOUT the extension
* method: 'ga' for Genetic Algorithm, 'ip' for Integer Programming
* h: help
* f: name of the file to save the output
* v: enable output verbosity
* t: time limit (in seconds) for the integer programming solver. Default value is 1800.
* m: value of μ (Size of the population). Default is 1000.
* l: value of λ (Number of individuals created by crossover). Must be even. Default value is 1000.
* p: value of ϕ (Likelihood of an individual to mutate). Default value is 0.25.
* o: value of ω (aximum number of generations without improvement). Default value is 500.

Remember that the program can only find instances that are inside the directory named Instances.
Also, it's necessary that the instance is formatted correctly and its extension is '.txt'.

## Examples

> $ python3 main.py tba1 ga -o 1000 -f tba1_results  
> $ python3 main.py tba10 ip -f tba10_ip_results -v -t 200  
> $ python3 main.py tba2 ga -m 10 -l 10 -p 0.8 -o 83

