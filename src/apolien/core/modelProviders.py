import ollama
from anthropic import Anthropic
import os

class OllamaProvider():

    def validate(self, model, config=None):
        ollama.chat(model, options=config)

    def generate(self, model, prompt, config=None):
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=config
        )
        return response['response']

class ClaudeProvider():

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide api_key parameter.")
        self.client = Anthropic(api_key=self.api_key)

    def validate(self, model, config=None):
        try:
            self.client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
        except Exception as e:
            raise ValueError(f"Failed to validate Claude model '{model}': {str(e)}")

    def generate(self, model, prompt, config=None):
        apiParams = self.mapConfig(config)

        if 'max_tokens' not in apiParams:
            apiParams['max_tokens'] = 4096

        try:
            response = self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **apiParams
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from Claude: {str(e)}")

    def mapConfig(self, config):
        # Map Ollama config options to Claude API parameters
        if not config:
            return {}

        apiParams = {}

        if 'temperature' in config:
            apiParams['temperature'] = config['temperature']

        if 'top_p' in config:
            apiParams['top_p'] = config['top_p']

        if 'top_k' in config:
            apiParams['top_k'] = config['top_k']

        # Handle num_predict -> max_tokens (Ollama uses -1 for unlimited)
        if 'num_predict' in config:
            num_predict = config['num_predict']
            if num_predict == -1:
                apiParams['max_tokens'] = 8192
            elif num_predict > 0:
                apiParams['max_tokens'] = num_predict

        if 'max_tokens' in config:
            apiParams['max_tokens'] = config['max_tokens']

        return apiParams

def getProvider(providerType, **kwargs):
    providerType = providerType.lower()

    if providerType == 'ollama':
        return OllamaProvider()
    elif providerType in ['claude', 'anthropic']:
        return ClaudeProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {providerType}. Supported types: 'ollama', 'claude'")
