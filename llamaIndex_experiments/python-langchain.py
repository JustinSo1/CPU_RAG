from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

if __name__ == '__main__':
    # Sets the current working directory to be the same as the file.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load environment file for secrets.
    try:
        if load_dotenv('../.env') is False:
            raise TypeError
    except TypeError:
        print('Unable to load .env file.')
        quit()

    # Define llm parameters
    llm = AzureChatOpenAI(
        deployment_name=os.environ['MODEL'],
        openai_api_version=os.environ['API_VERSION'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        azure_endpoint=os.environ['OPENAI_API_BASE'],
        openai_organization=os.environ['OPENAI_ORGANIZATION']
    )

    # Create Query
    messages = [
        ("system", "You are a helpful assistant.  Always say GO BLUE! at the end of your response."),
        ("human", "Explain step by step. Where is the University of Michigan?"),
    ]

    # Send a completion request.
    response = llm.invoke(messages)

    # Get and print response
    response = llm.invoke(messages)
    print(response.content)
