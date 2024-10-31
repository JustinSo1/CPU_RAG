import multiprocessing
from pprint import pprint

from llama_cpp import Llama


class LlamaCpp:
    def __init__(self, n_threads):
        self.llama = Llama.from_pretrained(
            repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename="*Q6_K_L.gguf",
            chat_format="llama-2",
            verbose=False,
            n_ctx=4096,
            n_threads=n_threads
        )

    def generate_text(self, prompt, *args, **kwargs):
        """Generate text using the cpp tokenizer and model"""
        output_text = self.llama(
            prompt=prompt,  # Prompt
            max_tokens=2048,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
            echo=True  # Echo the prompt back in the output
        )

        return [output_text['choices'][0]['text']]


if __name__ == '__main__':
    llm = LlamaCpp()
    print("Hello")
    print(llm.generate_text("Q: Who was the 16th president of the United States? A:"))

# llm = Llama.from_pretrained(
#     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
#     filename="*Q6_K_L.gguf",
#     # chat_format="chat_template.default",
#     verbose=True
# )
# output = llm(
#     "Q: Who was the 16th president of the USA? A: ",  # Prompt
#     max_tokens=None,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
#     stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
#     echo=True  # Echo the prompt back in the output
# )  # Generate a completion, can also call create_completion
# pprint(output)
