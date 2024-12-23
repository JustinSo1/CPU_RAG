from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.llama_cpp import LlamaCPP
from enum import Enum


class LLMName(Enum):
    GEMMA2_2B_IT = 1
    GPT_4o = 2


def get_documents(df):
    documents = [
        Document(
            text=row['passage'],
            # metadata={
            #     'ID': row['ID'],
            #     'Type': row['Type']
            # }
        )
        for _, row in df.iterrows()
    ]
    # print(documents)
    return documents


def split_documents(documents, embed_model):
    splitter = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=0,
        paragraph_separator="\n\n"
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    # splitter = SemanticSplitterNodeParser.from_defaults(embed_model=embed_model)
    # nodes = splitter.build_semantic_nodes_from_documents(documents, show_progress=True)
    return nodes


def construct_index(text_corpus, embed_model, llama_index_embeddings_path, vector_store):
    print("Building index")
    documents = get_documents(text_corpus)
    nodes = split_documents(documents, embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True, storage_context=storage_context)

    index.storage_context.persist(persist_dir=llama_index_embeddings_path)

    print("Built index")
    return index


def create_llama_cpp_model(model_url):
    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        # model_path="../data/models/gemma2/gemma-2-2b-it.Q5_K_M.gguf",
        temperature=0.1,
        max_new_tokens=256,
        context_window=8192,
        # kwargs to pass to __call__()
        # generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"repeat_penalty": 1, "penalize_nl": False, "chat_format": "gemma"},
        # transform inputs into Llama2 format
        # messages_to_prompt=messages_to_prompt,
        # completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    return llm
