from ..core import testsettings as settings
from ..core import customlogger as cl
from ..statistics import stats
from ..core import utils
from alive_progress import alive_bar
def faithfulness(logger, modelName, modelConfig, testsConfig, fileName, datasets, provider):
    try:
        lookback = testsConfig['cot_lookback']
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
    
    testedDatasets = []
    
    for datasetName in datasets:
        if datasetName not in settings.faithfulnessDatasets:
            continue
        testedDatasets.append(datasetName)
        
        dataset= utils.getLocalDataset(datasetName)
        
        with alive_bar(len(dataset), title=datasetName) as bar: 
            for questionNumber, question in enumerate(dataset):
                bar()
                if cl.isLoggingEnabled(logger):
                    cl.setLogfile(logger, str(f"faithfulness/{modelName}/{datasetName + str(questionNumber+1).zfill(3)}.log"), deleteExisting=True)
                
                
                prompt = utils.promptBuilder(settings.faithfulnessQuestionPrompt, question)
                
                logger.debug(f"\nPrompt:\n{prompt}")
                
                responseText = provider.generate(
                                            model=modelName,
                                            prompt=prompt,
                                            config=modelConfig
                                        )

                reasoning = utils.faithfulnessParseResponseText(responseText)
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
                    
                    reasoningPrompt = utils.promptBuilder(settings.faithfulnessContinuingPrompt, question, steps)
                    reasoningResponseText = provider.generate(
                                                        model=modelName,
                                                        prompt=reasoningPrompt,
                                                        config=modelConfig
                                                        )

                    lookbackAnswer = utils.faithfulnessParseAnswerString(reasoningResponseText)
                    
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

    cl.setLogfile(logger, fileName, indentPrefix="│  ")
    
    stats.generateAndPrintFaithfulnessReport(logger, differentAnswers, sameAnswers, tossedAnswers, tossedQuestions, sameStages, differentStages, processedQuestions, testedDatasets, modelName)