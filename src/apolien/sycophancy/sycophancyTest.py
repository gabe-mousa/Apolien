import pprint
from ..core import testsettings as settings
from ..core import customlogger as cl
from ..statistics import stats
from ..core import utils

def sycophancy(logger, modelName, modelConfig, testsConfig, fileName, datasets, provider):
    
    for dataset in datasets: 
        if dataset not in settings.sycophancyDatasets:
            continue
        dataset = utils.getLocalDataset(dataset)
        for row in dataset:
            pprint.pprint(row)