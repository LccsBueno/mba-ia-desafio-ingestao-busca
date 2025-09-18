import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():

    docs = PyPDFLoader(str(os.getenv("PDF_PATH"))).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,   
    ).split_documents(docs)

    if not splits:
        raise SystemExit(0)

    enriquecimento = [
        Document(
            page_content=doc.page_content,
            metadata={
                k : v for k, v in doc.metadata.items() if v not in ("", None)
            }
        )
        for doc in splits
    ]

    ids = [f"doc-{i}" for i in range (len(enriquecimento))]

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriquecimento, ids=ids)


if __name__ == "__main__":
    ingest_pdf()