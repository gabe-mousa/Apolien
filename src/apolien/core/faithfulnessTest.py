from . import testsettings as settings
from . import utils
from . import customlogger as cl
from ..statistics import stats
from . import utils

def faithfulness(logger, modelName, modelOptions, testConfig, fileName, datasets, provider):
    try:
        lookback = testConfig['cot_lookback']
    except:
        lookback = None
    
    #Set the variable needed for returning stats
    differentAnswers = 0
    sameAnswers = 0
    tossedAnswers = 0
    tossedQuestions = 0
    processedQuestions = 0
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
            
            logger.debug(f"\nPrompt:\n{prompt}")
            
            responseText = provider.generate(
                                        model=modelName,
                                        prompt=prompt,
                                        config=modelOptions
                                    )

            reasoning = utils.parseResponseText(responseText)
            reasoningSteps = reasoning["steps"]
            mainAnswer = reasoning['answer']
            
            logger.debug(f"\nResponse:\n\n{responseText}\n----------------------------Beginning CoT Analysis----------------------------\n\nParsed Steps and Answer:\n\n{reasoningSteps}\nAnswer: {mainAnswer}\n\n========================================================")
            
            if not reasoningSteps or not mainAnswer or mainAnswer == "None":
                tossedQuestions += 1
                tossedAnswers += len(reasoningSteps)
                continue
            
            processedQuestions += 1
            
            if not lookback:
                lookback = len(reasoningSteps)
            
            for i in range(lookback):
                if not reasoningSteps[:-lookback+i]:
                    continue
                
                steps = reasoningSteps[:-lookback+i]
                step = utils.interveneReasoningStep(steps[-1])
                steps[-1] = step
                
                reasoningPrompt = utils.promptBuilder(settings.continueFromReasoningFormatPrompt, question, steps)
                reasoningResponseText = provider.generate(
                                                    model=modelName,
                                                    prompt=reasoningPrompt,
                                                    config=modelOptions
                                                    )

                lookbackAnswer = utils.parseAnswerString(reasoningResponseText)
                
                if not lookbackAnswer:
                    tossedAnswers += 1
                    continue
                elif lookbackAnswer == mainAnswer:
                    sameAnswers += 1
                    sameStages[int(i/(lookback/3))] += 1
                else:
                    differentStages[int(i/(lookback/3))] += 1
                    differentAnswers += 1
                    
                logger.debug(f"Prompt:\n\n{reasoningPrompt}\n\nResponse:\n\n{reasoningResponseText}\n\nParsing Answer: {lookbackAnswer}\n========================================================")

    cl.setLogfile(logger, fileName)
    
    stats.generateAndPrintFaithfulnessReport(logger, differentAnswers, sameAnswers, tossedAnswers, tossedQuestions, sameStages, differentStages, processedQuestions, datasets, modelName)