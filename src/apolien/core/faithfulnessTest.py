import ollama
from . import testsettings as settings
from . import utils
from . import customlogger as cl
from ..statistics import stats
from . import utils

def faithfulness(logger, modelName, modelOptions, testConfig, fileName, datasets):
    try:
        lookback = testConfig['cot_lookback']
    except:
        lookback = None
    
    #Set the variable needed for returning stats
    differentAnswers = 0
    sameAnswers = 0
    tossedAnswers = 0
    tossedQuestions = 0
    sameStages = {0: 0,
                  1: 0,
                  2: 0}
    differentStages = {0: 0,
                  1: 0,
                  2: 0}
    
    for datasetName in datasets:
        dataset = utils.getLocalDataset(datasetName)
    
        for questionNumber, question in enumerate(dataset):
            
            if cl.isLoggingEnabled(logger):
                cl.setLogfile(logger, str(f"faithfulness/{modelName}/{datasetName + str(questionNumber+1).zfill(3)}.log"))
            
            
            prompt = utils.promptBuilder(settings.answerFormatPrompt, question)
            
            logger.debug("\nPrompt:\n")
            logger.debug(prompt)
            
            response = ollama.generate(
                                        model=modelName,
                                        prompt=prompt,
                                        options=modelOptions
                                    )
            
            logger.debug("\nResponse:\n")
            logger.debug(response["response"])
            
            logger.debug("----------------------------Beginning CoT Analysis----------------------------")
            
            reasoning = utils.parseResponseText(response['response'])
            reasoningSteps = reasoning["steps"]
            mainAnswer = reasoning['answer']
            
            if not reasoningSteps or not mainAnswer:
                tossedQuestions += 1
                tossedAnswers += len(reasoningSteps)
                continue
            
            logger.debug("\nParsed Steps and Answer:\n")
            logger.debug(reasoningSteps)
            logger.debug(str("Answer: " + mainAnswer + "\n"))
            logger.debug("========================================================")
            
            if not lookback:
                lookback = len(reasoningSteps)
            
            for i in range(lookback):
                if not reasoningSteps[:-lookback+i]:
                    continue
                
                steps = reasoningSteps[:-lookback+i]
                step = utils.interveneReasoningStep(steps[-1])
                steps[-1] = step
                
                reasoningPrompt = utils.promptBuilder(settings.continueFromReasoningFormatPrompt, question, steps)
                reasoningResponse = ollama.generate(
                                                    model=modelName,
                                                    prompt=reasoningPrompt,
                                                    options=modelOptions
                                                    )
                
                lookbackAnswer = utils.parseAnswerString(reasoningResponse['response'])
                
                if not lookbackAnswer or lookbackAnswer.strip() == "" or lookbackAnswer == "None":
                    tossedAnswers += 1
                elif lookbackAnswer == mainAnswer:
                    sameAnswers += 1
                    sameStages[int(i/(lookback/3))] += 1
                else:
                    differentStages[int(i/(lookback/3))] += 1
                    differentAnswers += 1
                    
                logger.debug("Prompt:\n")
                logger.debug(reasoningPrompt)
                logger.debug("\nResponse:\n")
                logger.debug(reasoningResponse['response'])
                logger.debug("\n")
                logger.debug(str("Parsing Answer: " + lookbackAnswer))
                logger.debug("========================================================")

    cl.setLogfile(logger, fileName)
    
    stats.generateAndPrintFaithfulnessReport(logger, differentAnswers, sameAnswers, tossedAnswers, tossedQuestions, sameStages, differentStages, datasets, modelName)