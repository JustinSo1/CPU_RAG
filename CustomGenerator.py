from target_benchmark.generators import AbsGenerator
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

class CustomGenerator(AbsGenerator):
    def __init__(self):
        # Set the current working directory to the directory of this script
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Load environment file for secrets
        try:
            if not load_dotenv('target/.env'):
                raise TypeError
        except TypeError:
            print('Unable to load .env file.')
            quit()

        # Initialize the AzureOpenAI client
        self.client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            api_version=os.environ['API_VERSION'],
            azure_endpoint=os.environ['OPENAI_API_BASE'],
            organization="016732"
        )

        self.model = "gpt-4o"
        print(f"Initializing generator with model: {self.model}")

    # Generate the answer to the query
    def generate(self, table_str: str, query: str) -> str:
        # Combine the table string and query to form the input for the model
        input_text = f"Table: {table_str}\nQuestion: {query}\nAnswer:"
        print(f"Generating answer for input: {input_text}")

        messages=[
            {"role": "system","content": "You are a QA chatbot"},
            {"role": "user","content": input_text},
        ]

        # Use the loaded generator model
        outputs = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stop=None
        )

        # Extract the answer from the response
        answer = outputs.choices[0].message.content
        return answer