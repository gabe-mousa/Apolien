# To be able to run these example files without Apolien installed through pip
# you can run these files in `editable` mode. To do so navigate in your CLI to
# the main directory Apolien/ and run `pip install -e ./`. This will install 
# apolien locally and now you should be able to run this file. 
import apolien as apo 

allTests = ['cot_faithfulness', 'sycophancy']
datasetsTest = ['faithfulness_math_one', 'faithfulness_math_five','sycophancy_30']
models = ["llama3.2:1b","smallthinker","deepseek-r1:1.5b"]

def testAllModels(models, tests):
    for modelName in models:
        model = apo.evaluator(
            modelName,
            {
                "temperature": 0.8,
                "num_predict": -1
            },
            fileLogging=True,
            fileName=f"{modelName}.log"
        )
        model.evaluate(
            userTests=tests,
            testsConfig={},
            datasets=datasetsTest,
            testLogFiles=True
        )


def testModel(modelName, tests):
    model = apo.evaluator(
        modelName,
        {
            "temperature": 0.8,
            "num_predict": -1
        },
        fileLogging=True,
        fileName=f"{modelName}.log"
    )
    model.evaluate(
        userTests=tests,
        testsConfig={},
        datasets=datasetsTest,
        testLogFiles=True
    )

testModel(models[0], ['sycophancy'])
#testAllModels(models, ['sycophancy'])