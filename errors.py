class DatasetSizeSmallerThenBatchSize(Exception):
    def __init__(self, message):
        super().__init__(message)

class TheRemainderOfTheDivision(Exception):
    def __init__(self, message):
        super().__init__(message)

class TheQuotientIsNotDividedIntoTwo(Exception):
    def __init__(self, message):
        super().__init__(message)