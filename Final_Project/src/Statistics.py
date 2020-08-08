class Statistics:
    def __init__(self, instanceName, method):
        self.instanceName = instanceName
        self.method = method
        self.runningTime = 0
        self.objValue = 0
        self.firstGenerationObjValue = 0
        self.numberOfGenerations = 0
        self.optimalityGap = 0

