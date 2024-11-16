import multiprocessing
import subprocess
import sys
import re
import io
from pygemma import Gemma, ModelType, ModelTraining


class GemmaCPP:
    """Wrapper for the C++ implementation of Gemma"""

    def __init__(self, gemma_cpp, tokenizer, compressed_weights, model):
        self.gemma_cpp = gemma_cpp
        self.tokenizer = tokenizer
        self.compressed_weights = compressed_weights
        self.model = model

    def eliminate_long_dots(self, input_string):
        """Eliminate long sequences of dots from the input string"""
        # Define a regular expression pattern to match sequences of 2 or more dots
        pattern = r'\.{2,}'

        # Replace all occurrences of the pattern with a space
        output_string = re.sub(pattern, ' ', input_string)

        return output_string.strip()

    def beautify_string(self, input_string):
        """Clean the input string by removing non-letter characters at the beginning
           and isolated letters at the end after multiple spaces"""
        # Remove non-letter characters at the beginning of the string
        output_string = re.sub(r'^[^a-zA-Z]+', '', input_string.strip())

        # Remove isolated letters at the end of the output string after multiple spaces
        output_string = re.sub(r'\s{3,}(.+)\Z', '', output_string.strip())

        return output_string

    def generate_text(self, prompt, *args, **kwargs):
        """Generate text using the cpp tokenizer and model"""

        # Define the shell command
        prompt = prompt.replace('"', '').replace("'", "").replace("`", "")
        #         print("Gen text", prompt)
        shell_command = f'echo "{prompt}" | {self.gemma_cpp} -- --tokenizer {self.tokenizer} --compressed_weights {self.compressed_weights} --model {self.model} --verbosity 0'

        # Execute the shell command and redirect stdout to the Python script's stdout
        process = subprocess.Popen(shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        output_text = []
        #         output_text = ''
        reading_block = "[ Reading prompt ]"

        #         # Communicate with the process and capture stdout
        #         for k, char in enumerate( iter(lambda: process.stdout.read(1), b'') ):
        #             single_char = char.decode(sys.stdout.encoding)
        #             output_text += single_char
        #             if len(output_text) % 20 == 0:
        #                 count_reading_blocks = output_text.count(reading_block)
        #                 if count_reading_blocks > 1:
        #                     break
        for line in io.TextIOWrapper(process.stdout, encoding=sys.stdout.encoding):
            output_text.append(line)

        output_text = ''.join(output_text)
        # Remove long sequences of dots and the reading block, beautify the string
        output_text = output_text.replace(reading_block, "")
        output_text = self.eliminate_long_dots(output_text)
        output_text = self.beautify_string(output_text)
        output_text = prompt + output_text

        # Return output text
        return [output_text]


class GemmaCPPPython:
    """Wrapper for the Python Wrapper for Gemma.cpp"""

    def __init__(self, tokenizer, compressed_weights, n_threads=max(multiprocessing.cpu_count() - 2, 1),
                 model_type=ModelType.Gemma2B,
                 model_training=ModelTraining.GEMMA_IT):
        #         self.gemma_cpp = gemma_cpp
        #         self.tokenizer = tokenizer
        #         self.compressed_weights = compressed_weights
        #         self.model = model
        self.gemma = Gemma(
            tokenizer_path=tokenizer,
            compressed_weights_path=compressed_weights,
            model_type=model_type,
            model_training=model_training,
            # n_threads=n_threads
        )

    def generate_text(self, prompt, *args, **kwargs):
        """Generate text using the cpp tokenizer and model"""
        output_text = self.gemma(prompt=prompt, max_tokens=4096, max_generated_tokens=512)

        return [output_text]
