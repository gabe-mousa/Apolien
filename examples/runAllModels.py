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
testsRun = ['sycophancy']

runConfig = {
    claude_haiku : [['sycophancy_1000'], 'claude'],
    claude_sonnet: [['sycophancy_1000'], 'claude'],
    "llama3.2:1b": [['sycophancy_100'], 'ollama'],
    "deepseek-r1:1.5b" : [['sycophancy_100'], 'ollama'],
    'smallthinker' : [['sycophancy_30'], 'ollama']
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

for model in runConfig:
    testModel(model, runConfig[model][0], runConfig[model][1])
