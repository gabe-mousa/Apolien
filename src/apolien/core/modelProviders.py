import ollama
from anthropic import Anthropic
from openai import OpenAI
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
            # Build validation parameters dynamically from config
            validation_params = {
                'model': model,
                'max_tokens': 10,
                'messages': [{"role": "user", "content": "Hi"}]
            }
            
            # Add all parameters from config dynamically
            if config:
                for param_name in config:
                    validation_params[param_name] = config[param_name]
            
            self.client.messages.create(**validation_params)
        except Exception as e:
            raise ValueError(f"Failed to validate Claude model '{model}': {str(e)}")

    def generate(self, model, prompt, config=None):
        # Build generation parameters dynamically from config
        params = {
            'model': model,
            'messages': [{"role": "user", "content": prompt}]
        }
        
        # Set default max_tokens if not provided
        if not config or 'max_tokens' not in config:
            params['max_tokens'] = 4096
        
        # Add all parameters from config dynamically
        if config:
            for param_name in config:
                params[param_name] = config[param_name]
        
        try:
            response = self.client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from Claude: {str(e)}")

class OpenAIProvider():

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
        self.client = OpenAI(api_key=self.api_key)

    def validate(self, model, config=None):
        try:
            # Build validation parameters dynamically from config
            validation_params = {
                'model': model,
                'input': "Hi"
            }
            
            if config:
                for param_name in config:
                    validation_params[param_name] = config[param_name]
            
            self.client.responses.create(**validation_params)
        except Exception as e:
            raise ValueError(f"Failed to validate OpenAI model '{model}': {str(e)}")

    def generate(self, model, prompt, config=None):
        params = {
                'model': model,
                'input': prompt
            }
            
        if config:
            for param_name in config:
                params[param_name] = config[param_name]
        try:
            response = self.client.responses.create(**params)
            return response.output_text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from OpenAI: {str(e)}")
        

def getProvider(providerType, **kwargs):
    providerType = providerType.lower()

    if providerType == 'ollama':
        return OllamaProvider()
    elif providerType in ['claude', 'anthropic']:
        return ClaudeProvider(**kwargs)
    elif providerType in ['openai', 'gpt']:
        return OpenAIProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {providerType}. Supported types: 'ollama', 'claude', 'openai'")
