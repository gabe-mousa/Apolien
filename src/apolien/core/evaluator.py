import ollama
from . import constants
from . import testsettings
from . import customlogger as cl
import logging

class evaluator():
    """Main class for Apolien, holds model information and used to call other testing procedures"""
    
    def __init__(self, 
                 model: str, 
                 modelConfig: dict | None = None, 
                 statsConfig: list[str] |  None = None,
                 fileLogging: bool = False,
                 fileName: str = testsettings.outputFile):
        
        ollama.chat(model, options=modelConfig) # Validating the model name and configs
        self.modelName = model
        self.modelConfig = modelConfig
        self.statsConfig = statsConfig
        self.logger = cl.setupLogger(toFile=fileLogging, filename=fileName)
        self.outfile = fileName
        self.testsConfig = {
            "cot_lookback" : None
        }
        print("Apolien Initialized")
    
    def evaluate(self, 
                 userTests: list[str],
                 testsConfig: dict = {},
                 fileName: str | None = None,
                 testLogFiles: bool = False,
                 datasets: list = ['math_debug_five']):
        try:
            if not fileName:
                fileName = self.outfile
            
            self.testsConfig.update(testsConfig)
            
            if testLogFiles:
                self.logger.setLevel(logging.DEBUG)
            else:
                self.logger.setLevel(logging.INFO)

            for test in userTests:
                print("Starting",test,"tests")
                
                constants.testMapping[test](self.logger, self.modelName, self.modelConfig, self.testsConfig, self.outfile, datasets)
                
                print("Finished",test,"tests")
        except Exception as err:
            raise err
