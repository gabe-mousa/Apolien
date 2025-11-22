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
testsRun = ['sycophancy', 'cot_faithfulness']

runConfigFull = {
    claude_haiku : [['sycophancy_1000'], 'claude'],
    claude_sonnet: [['sycophancy_1000'], 'claude'],
    "llama3.2:1b": [['sycophancy_100'], 'ollama'],
    "deepseek-r1:1.5b" : [['sycophancy_100'], 'ollama'],
    'smallthinker' : [['sycophancy_30'], 'ollama'],
    'gpt-5-nano' : [['sycophancy_1000'], 'openai']
}

runConfigDebug = {
    claude_haiku : [['debug_math_1', 'sycophancy_1'], 'claude'],
    claude_sonnet: [['debug_math_1', 'sycophancy_1'], 'claude'],
    "llama3.2:1b": [['debug_math_1', 'sycophancy_1'], 'ollama'],
    "deepseek-r1:1.5b" : [['debug_math_1', 'sycophancy_1'], 'ollama'],
    'gpt-5-nano' : [['debug_math_1', 'sycophancy_1'], 'openai']
}

os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")

def testModel(modelName, dataset, provider):
    evaluator = apo.evaluator(
        model=modelName,
        modelConfig={
            "temperature": 1
        },
        provider=provider,
        fileLogging=True,
        fileName=f"{modelName}.log"
    )

    evaluator.evaluate(
        userTests=testsRun,
        datasets=dataset,
        testLogFiles=False
    )

for model in runConfigDebug:
    testModel(model, runConfigDebug[model][0], runConfigDebug[model][1])
