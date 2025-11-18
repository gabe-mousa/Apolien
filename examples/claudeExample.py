# To be able to run these example files without Apolien installed through pip
# you can run these files in `editable` mode. To do so navigate in your CLI to
# the main directory Apolien/ and run `pip install -e ./`. This will install 
# apolien locally and now you should be able to run this file. 

#For a list of available claude models, access: https://docs.claude.com/en/docs/about-claude/models/overview#legacy-models
import os
import apolien as apo
from dotenv import load_dotenv

#Load env variables
load_dotenv()

claude_haiku = 'claude-haiku-4-5'
claude_sonnet = "claude-sonnet-4-5"
models = [claude_haiku, claude_sonnet]
modelName = claude_haiku
datasetTest = ['debug_math_1', 'sycophancy_10']
testsRun = ['sycophancy']
# Example 1: Using Claude with environment variable for API key
# Set ANTHROPIC_API_KEY environment variable before running
def testOSKey():
    os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
    eval_claude = apo.evaluator(
        model=modelName,
        modelConfig={
            "temperature": 0.8,
            "max_tokens": 4096
        },
        provider="claude",
        fileLogging=True,
        fileName=f"{modelName}.log"
    )

    eval_claude.evaluate(
        userTests=testsRun,
        datasets=datasetTest,
        testLogFiles=True
    )

# Example 2: Using Claude with API key passed directly
# Uncomment to use:
def testParameterAPIKey():
    eval_claude = apo.evaluator(
        model=modelName,
        modelConfig={
            "temperature": 0.8,
            "max_tokens": 4096
        },
        provider="claude",
        api_key="your-api-key-here",
        fileLogging=True,
        fileName=f"{modelName}.log"
    )

    eval_claude.evaluate(
        userTests=testsRun,
        datasets=datasetTest,
        testLogFiles=True
    )

testOSKey()
# for model in models[0]: 
#     modelName = model
#     testOSKey()
    #testParameterAPIKey()