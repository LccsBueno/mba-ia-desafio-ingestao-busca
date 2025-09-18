from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate

import os 
from dotenv import load_dotenv

load_dotenv()

question_template = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    CONTEXTO:
    {context}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
      "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {query}

    RESPONDA A "PERGUNTA DO USUÁRIO"
    """
)

# Refactored for RunnableLambda compatibility
def extract_context_chain_input(inputs):
    query = inputs["query"]
    results = inputs["results"]
    return {
        "query": query,
        "context": "".join([getattr(doc, "page_content", "") for doc, score in results])
    }

def search_prompt(query: str):
    
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )
    
    results = store.similarity_search_with_score(query, k=10)

    model = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_CHAT"), temperature=1)
    
    # Prepare the chain with the refactored extract_context_chain_input
    chain = RunnableLambda(extract_context_chain_input) | question_template | model

    # Pass both query and results as input
    return chain.invoke({"query": query, "results": results})
    
    # chain = RunnableLambda(extract_context) | question_template | model
    
    # return chain.invoke({"query": query})
    ''