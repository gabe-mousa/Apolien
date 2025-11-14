import Apolien.src.apolien as apo
import os

# Example 1: Using Claude with environment variable for API key
# Set ANTHROPIC_API_KEY environment variable before running

eval_claude = apo.evaluator(
    model="claude-3-5-sonnet-20241022",
    modelConfig={
        "temperature": 0.8,
        "max_tokens": 4096
    },
    provider="claude",
    fileLogging=True
)

eval_claude.evaluate(
    userTests=['cot_faithfulness'],
    datasets=['math_debug_five'],
    testLogFiles=False
)

# Example 2: Using Claude with API key passed directly
# Uncomment to use:

# eval_claude = apo.evaluator(
#     model="claude-3-5-sonnet-20241022",
#     modelConfig={
#         "temperature": 0.8,
#         "max_tokens": 4096
#     },
#     provider="claude",
#     api_key="your-api-key-here",
#     fileLogging=True
# )
#
# eval_claude.evaluate(
#     userTests=['cot_faithfulness'],
#     datasets=['simple_math_100'],
#     testLogFiles=False
# )

# Example 3: Testing different Claude models

# Claude 3 Opus (most capable)
# eval_opus = apo.evaluator(
#     model="claude-3-opus-20240229",
#     provider="claude",
#     fileLogging=True
# )

# Claude 3.5 Sonnet (balanced performance)
# eval_sonnet = apo.evaluator(
#     model="claude-3-5-sonnet-20241022",
#     provider="claude",
#     fileLogging=True
# )

# Claude 3 Haiku (fast and efficient)
# eval_haiku = apo.evaluator(
#     model="claude-3-haiku-20240307",
#     provider="claude",
#     fileLogging=True
# )
