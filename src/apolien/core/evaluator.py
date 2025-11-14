from . import constants
from . import testsettings
from . import customlogger as cl
from . import modelProviders
import logging

class evaluator():
    """Main class for Apolien, holds model information and used to call other testing procedures"""
    
    def __init__(self,
                 model: str,
                 modelConfig: dict | None = None,
                 statsConfig: list[str] |  None = None,
                 fileLogging: bool = False,
                 fileName: str = testsettings.outputFile,
                 provider: str = 'ollama',
                 api_key: str | None = None):
        """
        Initialize the Apolien evaluator.

        Args:
            model: Model name (e.g., 'llama2' for Ollama or 'claude-3-5-sonnet-20241022' for Claude)
            modelConfig: Model configuration options (temperature, max_tokens, etc.)
            statsConfig: Statistics configuration
            fileLogging: Whether to enable file logging
            fileName: Output file name for logs
            provider: LLM provider to use ('ollama' or 'claude'). Defaults to 'ollama'.
            api_key: API key for the provider (required for Claude, uses ANTHROPIC_API_KEY env var if not provided)
        """

        # Get the appropriate provider
        provider_kwargs = {}
        if api_key:
            provider_kwargs['api_key'] = api_key

        self.provider = modelProviders.getProvider(provider, **provider_kwargs)

        # Validate the model name and configs
        self.provider.validate(model, modelConfig)

        self.modelName = model
        self.modelConfig = modelConfig
        self.statsConfig = statsConfig
        self.logger = cl.setupLogger(toFile=fileLogging, filename=fileName)
        self.outfile = fileName
        self.testsConfig = {
            "cot_lookback" : None
        }
        print(f"Apolien Initialized: {self.modelName} (provider: {provider})")
    
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

                constants.testMapping[test](self.logger, self.modelName, self.modelConfig, self.testsConfig, self.outfile, datasets, self.provider)

                print("Finished",test,"tests")
        except Exception as err:
            raise err
