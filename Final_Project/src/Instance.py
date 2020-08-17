import os


class Instance:
    """
    An instance of the TB problem

    ...

    Arguments
    ----------
    instanceFile: path for the file of an instance

    Attributes
    ----------
    n : int
        number of tasks
    m : int
        number of operators
    p : 2-dimensional matrix of integers
        p[i][j] gives the time needed for the operator j to end task i

    """

    def __init__(self, instanceFile):
        self.name = os.path.basename(instanceFile)
        self.n, self.m, self.p = self.read_instance_file(instanceFile)

    @staticmethod
    def read_instance_file(instanceName):
        with open('../Instances/{0}.txt'.format(instanceName), 'r') as file:
            lines = file.readlines()

            n = int(lines[0].split()[0])
            m = int(lines[1].split()[0])

            p = [[0 for i in range(m)] for j in range(n)]
            for j in range(m):
                row = lines[5+j].split()
                for i in range(n):
                    p[i][j] = float(row[i])

            return n, m, p
