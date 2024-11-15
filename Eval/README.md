# Run the code
python -m venv <name>
source <name>/bin/activate
pip3 install -r requirements.txt
python3 evaulation.py

## Environment File in the following Way
MODEL=gpt-4o
OPENAI_API_BASE=<API_BASE>
OPENAI_API_KEY=<API_KEY>
API_VERSION=2024-06-01
OPENAI_ORGANIZATION=<API_ORGNIAZATION>

## Organization
- Stuff saved in response/<file_name>
- Need to change *file_name* in evaluation.py to what you want to evaluate
- Basic data output to command line