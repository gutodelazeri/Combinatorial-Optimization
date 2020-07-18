import argparse
import IPSolver as solver


def main():
    parser = argparse.ArgumentParser(description='INF05010 - Final Project')
    parser.add_argument('outputFile', help='Name of the file to store the solution')
    parser.add_argument('inputFile', help='Nme of the file containing the input instance')
    parser.add_argument('method', help='Name of the file to store the solution')
    args = parser.parse_args()


if __name__ == "__main__":
    s = solver.IPSolver('tba1', 5, True)
    s.solveModel()
