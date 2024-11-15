from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from datasets import load_dataset
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper


def main():
    # template = """Question: {question}
    #
    # Answer: Let's work this out in a step by step way to be sure we have the right answer."""
    # prompt = PromptTemplate.from_template(template)
    # Callbacks support token-wise streaming
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="../data/gemma2/2b_it_v2.gguf",
        n_ctx=1024,
        # temperature=0.75,
        # max_tokens=4096,
        # top_p=1,
        # callback_manager=callback_manager,
        # verbose=True,  # Verbose is required to pass to the callback manager
        # repeat_penalty=1.0
    )
    # question = """
    # Question: A rap battle between Stephen Colbert and John Oliver
    # """
    # print(llm.invoke(question))

    dataset = load_dataset(
        "explodinggradients/amnesty_qa",
        "english_v3",
        trust_remote_code=True
    )
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset["eval"])
    evaluator_llm = LangchainLLMWrapper(llm)

    metrics = [
        LLMContextRecall(llm=evaluator_llm),
        FactualCorrectness(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        # SemanticSimilarity(embeddings=evaluator_embeddings)
    ]
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    df = results.to_pandas()
    print(df.head())


if __name__ == '__main__':
    main()
