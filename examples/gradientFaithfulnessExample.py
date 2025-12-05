#!/usr/bin/env python3
"""
Example demonstrating gradient faithfulness evaluation in Apolien

This example shows how to use the new continuous scoring system
for chain-of-thought faithfulness testing.
"""

from apolien import evaluator
from dotenv import load_dotenv
import os
load_dotenv()

def main():
    os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
    # Initialize evaluator with a local model
    eval_instance = evaluator(
        model="claude-haiku-4-5",
        provider='claude',
        fileLogging=True,
        fileName="claude-haiku-4-5.log"
    )

    testConfig = {
        'cotFaithfulness': {
            'enableGradientScoring': True,     # Enable gradient scoring
            'severityLevels': ['minor', 'moderate', 'major'],  # Intervention types
            'correlationMethod': 'pearson'     # Correlation calculation method
        },
        'cot_lookback': 3  # Look back at last 3 reasoning steps
    }
    
    eval_instance.evaluate(
        userTests=['cot_faithfulness'],
        testsConfig=testConfig,
        datasets=['debug_math_5'],
        testLogFiles=True
    )

if __name__ == "__main__":
    main()