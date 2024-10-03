from datasets import load_dataset

if __name__ == '__main__':
    text_corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    print(text_corpus)

    question_answer = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    print(question_answer)
