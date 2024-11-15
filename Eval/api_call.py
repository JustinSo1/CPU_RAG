from openai import AzureOpenAI
import os
from dotenv import load_dotenv

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"[FEEDBACK]: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

class API_CLIENT():
  def __init__(self):
    #Sets the current working directory to be the same as the file.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #Load environment file for secrets.
    try:
        if load_dotenv('.env') is False:
            raise TypeError
    except TypeError:
        print('Unable to load .env file.')
        quit()

    #Create Azure client
    self.client = AzureOpenAI(
        api_key=os.environ['OPENAI_API_KEY'],
        api_version=os.environ['API_VERSION'],
        azure_endpoint = os.environ['OPENAI_API_BASE'],
        organization = "016732"
    )
    self.model = os.environ['MODEL']

  def send_query(self, inputPrompt, userAnswer, correctAnswer):
   # Format the prompt with the given instruction, response, and reference answer
    prompt = EVALUATION_PROMPT.format(
        instruction=inputPrompt,
        response=userAnswer,
        reference_answer=correctAnswer
    )

    #Create Query
    messages=[
            {"role": "system","content": "You are a fair evaluator language model"},
            {"role": "user","content": prompt},
        ]

    # Send a completion request.
    response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stop=None)

    #Print response.
    return response.choices[0].message.content