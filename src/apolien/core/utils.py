import os
import re
import random
from . import testsettings
import pandas as pd
import ast

def promptBuilder(*args) -> str:
    """Build a prompt with newlines between each argument, takes in any amount of arguments"""
    
    prompt = ""
    
    for arg in args:
        try:
            if isinstance(arg, str):
                prompt = prompt + arg + "\n\n"
            elif isinstance(arg, list):
                for val in arg:
                    prompt = prompt + val + "\n\n"
        except Exception as err:
            exceptStr = "Bad type passed into promptBuilder"
            raise Exception(exceptStr, str(err))
    
    return prompt

def parseResponseText(text: str) -> dict:
    """Take a body of text with a numbered list and a keyword answer, and parse out each of the steps in the numbered
    list and the answer."""
    
    # Trim spare text
    text = text.strip()
    
    answer = parseAnswerString(text)

    # Parse out for the numbered steps (standard format: "1. step text")
    steps = re.findall(
        r'(?:^|\n)\s*\d+\.\s*(.+?)(?=(?:\n\s*\d+\.|\Z))',
        text,
        re.DOTALL
    )

    # If no numbered steps found, try alternative format: "Step N: step text"
    if not steps:
        # tolerate common markdown wrappers around the step label, e.g. "**Step 1:**"
        # allow up to 3 emphasis/backtick/tilde chars before/after the 'Step' token
        steps = re.findall(
            r'(?im)(?:^|\n)\s*[*_`~]{0,3}\s*step\s+\d+\s*[*_`~]{0,3}\s*:\s*(.+?)(?=(?:\n\s*[*_`~]{0,3}\s*step\s+\d+\s*[*_`~]{0,3}\s*:|\Z))',
            text,
            re.DOTALL
        )

    # Clean all the steps and put in a list for returning
    cleanedSteps = [
    str(re.sub(r'(?i)\b(?:final\s*)?answer\b.*', '', step).strip())
    for step in steps
    ]
    return {"steps": cleanedSteps, "answer": str(answer) or None}

def parseAnswerString(text: str) -> str | None:
    """Parse out a keyword 'Answer: ' from a body of text, and return the answer"""
    # Try to find the answer 
    answerMatch = re.search(r'(?i)(?:^|\n)\s*(?:[*_`~]{1,3}\s*)*(?:final\s*)?(?:[*_`~]{1,3}\s*)?answer[^:\n]*[:\s]*([\s\S]*?)(?=\n\d+\.|\Z)', text)

    answer = None
    if answerMatch:
        answer = answerMatch.group(1).strip()
        # try parsing with basic regex for answer string
        answer = re.sub(r'^\s*(?:answer|final answer)\s*[:\-]?\s*', '', answer, flags=re.IGNORECASE).strip()

        # If the answer is in LaTeX boxed format (thanks deepseek)
        boxed = re.search(r'\\boxed\{([^}]*)\}', answer)
        if boxed:
            answer = boxed.group(1).strip()
        else:
            # Handle common wrappers around \boxed, \(\boxed{...}\), $\boxed{...}$, \[\boxed{...}\]
            boxed_math = re.search(r'\\\(\s*\\boxed\{([^}]*)\}\s*\\\)', answer) or \
                         re.search(r'\\\[\s*\\boxed\{([^}]*)\}\s*\\\]', answer) or \
                         re.search(r'\$\s*\\boxed\{([^}]*)\}\s*\$', answer)
            if boxed_math:
                answer = boxed_math.group(1).strip()

    # If an answer was not found, then parse natural language to try to find it
    if not answer:
        # Search phrases like "the answer is X" or "final result is X"
        matchNatural = re.search(
            r'(?i)(?:the\s+)?(?:final\s+)?(?:answer|result|output)\s*(?:is|=)\s*([A-Za-z0-9\.\-\s]+)', text
        )
        if matchNatural:
            answer = matchNatural.group(1).strip()

    # Cleaning answer string for digits
    if answer is not None:
        num_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', answer)
        if num_match:
            return num_match.group(0)

    return answer if answer is not None else None

def shiftNumbers(text: str) -> str:
    """Shift all numeric values in a reasoning step by a fixed or random integer. If no numbers are found, 
    returns the text unchanged.
    Example: "Add 3 and 4" -> "Add 5 and 6"
    """
    
    numbers = re.findall(r'\b\d+\b', text)
    if not numbers:
        return text  # No numbers to modify

    def replaceNum(match):
        num = int(match.group())
        offset = random.choice([-3, -2, -1, 1, 2, 3])
        return str(num + offset)

    return re.sub(r'\b\d+\b', replaceNum, text)

def reverseOperators(text: str) -> str:
    """Reverse all of the operators in a string of text."""
    
    replacements = {
        '+': '-', 
        '-': '+', 
        '*': '/', 
        '/': '*',
        'greater': 'less', 
        'less': 'greater',
        'more': 'less', 
        'increase': 'decrease',
        'decrease': 'increase', 
        'add': 'subtract',
        'subtract': 'add', 
        'multiply': 'divide',
        'divide': 'multiply',
        'plus': 'minus',
        'minus': 'plus'
    }

    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in replacements.keys()) + r')\b', flags=re.IGNORECASE)
    if not pattern.search(text):
        return text

    def replaceOp(match):
        op = match.group().lower()
        return replacements.get(op, op)

    return pattern.sub(replaceOp, text)


def negateConclusion(text: str) -> str:
    """Negate any conclusions found in an answer string."""
    
    found = False

    def negation(match):
        nonlocal found
        found = True
        return match.group(1) + "not " + match.group(2)

    text = re.sub(r'(?i)\b(the\s+result\s+is\s+)([^\.\n]+)', negation, text)
    text = re.sub(r'(?i)\b(the\s+answer\s+is\s+)([^\.\n]+)', negation, text)
    text = re.sub(r'(?i)\b(the\s+output\s+is\s+)([^\.\n]+)', negation, text)

    return text if found else text


def interveneReasoningStep(step: str, mode: int = 0) -> str:
    """Given a particular reasoning step, negate or reverse the words accordingly
    Example: '1. First, I'll multiply 3 and 10 together to get 30.' -> 'First, I'll divide 0 and 9 together to get 29.'
    """
    if mode == 1:
        return shiftNumbers(step)
    elif mode == 2:
        return reverseOperators(step)
    elif mode == 3:
        return negateConclusion(step)
    else:
        step = shiftNumbers(step)
        step = reverseOperators(step)
        step = negateConclusion(step)
        return step

def getLocalDataset(dataset: str) -> list:
    try:
        dataset = testsettings.datasets[dataset]
    except KeyError as err:
        raise KeyError(f"Apolien does not have a built-in dataset named: {dataset}")
    
    if isinstance(dataset, list):
        return dataset
    
    # Convert relative paths to be relative to the core directory where this file is located
    if isinstance(dataset, str) and not os.path.isabs(dataset):
        core_dir = os.path.dirname(os.path.abspath(__file__))
        datasetFile = os.path.join(core_dir, dataset)
    
    if not os.path.isfile(datasetFile):
        raise FileNotFoundError("Provided dataset does not exist")
    
    if dataset in testsettings.faithfulnessDatasets.values():
        return getFaithfulnessDataset(datasetFile)
    if dataset in testsettings.sycophancyDatasets.values():
        return getSycophancyDataset(datasetFile)
    
    raise Exception("Apolien does not yet support custom datasets")

def getFaithfulnessDataset(dataset: str): 
    file = open(dataset, 'r')
    
    fileData = file.read()
    data = fileData.splitlines()
    file.close()
    
    return data

def getSycophancyDataset(dataset: str):
    print(dataset)
    df = pd.read_csv(dataset)

    questions = []

    for _, row in df.iterrows():
    # Replace numpy array notation with lists
        choicesStr = row['choices']
        # Remove all whitespace/newlines that might be in the middle
        choicesStr = ' '.join(choicesStr.split())
        # Remove numpy array notation and dtype specification
        choicesStr = choicesStr.replace("array(", "[").replace(", dtype=object)", "]")
        # Handle the case where we have multiple array() calls (for labels and text)
        choicesStr = choicesStr.replace("array([", "[").replace("])", "]")
        
        # Now parse as a dictionary - handle the format: {'key': [...], 'key': [...]}
        choicesDict = ast.literal_eval(choicesStr)
        choicesList = choicesDict['text']
        
        questionObj = {
            'answer': row['answerKey'],
            'question': row['question'],
            'choices': choicesList[0]
        }
        
        questions.append(questionObj)

    return questions