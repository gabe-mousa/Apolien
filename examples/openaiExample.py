# To be able to run these example files without Apolien installed through pip
# you can run these files in `editable` mode. To do so navigate in your CLI to
# the main directory Apolien/ and run `pip install -e ./`. This will install 
# apolien locally and now you should be able to run this file. 

# For a list of available OpenAI models, access: https://platform.openai.com/docs/models
# This integration uses the OpenAI Responses API (released March 2024)
import os
import apolien as apo
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Available OpenAI models (common ones)
gpt_51 = 'gpt-5.1'
gpt_51_mini = 'gpt-5-mini'
gpt_5 = 'gpt-5'
gpt_5_nano = 'gpt-5-nano'

models = [gpt_51, gpt_51_mini, gpt_5, gpt_5_nano]
modelName = gpt_5_nano
datasetTest = ['debug_math_5', 'sycophancy_30']
testsRun = ['cot_faithfulness']

# Example 1: Using OpenAI with environment variable for API key
# Set OPENAI_API_KEY environment variable before running
def testOSKey():
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    eval_openai = apo.evaluator(
        model=modelName,
        modelConfig={
            "max_output_tokens": 4096
        },
        provider="openai",
        fileLogging=True,
        fileName=f"{modelName}.log"
    )

    eval_openai.evaluate(
        userTests=testsRun,
        datasets=datasetTest,
        testLogFiles=True
    )

# Example 2: Using OpenAI with API key passed directly
# Uncomment to use:
def testParameterAPIKey():
    eval_openai = apo.evaluator(
        model=modelName,
        modelConfig={
            "temperature": 0.8,
            "max_output_tokens": 4096
        },
        provider="openai",
        api_key="your-api-key-here",
        fileLogging=True,
        fileName=f"{modelName}.log"
    )

    eval_openai.evaluate(
        userTests=testsRun,
        datasets=datasetTest,
        testLogFiles=True
    )

testOSKey()
