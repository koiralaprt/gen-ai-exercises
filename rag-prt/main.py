import os
import sys

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.document_loaders.confluence import (
    ConfluenceLoader,
    ContentFormat,
)
from langchain_community.document_loaders.pdf import PDFMinerLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CONFLUENCE_API_KEY = os.environ.get("CONFLUENCE_API_KEY", None)
if not CONFLUENCE_API_KEY:
    print("error. CONFLUENCE_API_KEY env var not found.")

if len(sys.argv) != 2:
    print("usage: python main.py 'USER PROMPT HERE'")
    sys.exit(0)
user_prompt = sys.argv[1]

bedrock_embedder = BedrockEmbeddings()

pdf_file = PDFMinerLoader(file_path="./source/anti-harassment.pdf").load()
docx_file = Docx2txtLoader(file_path="./source/nomination.docx").load()
confluence = ConfluenceLoader(
    api_key=CONFLUENCE_API_KEY,
    cloud=True,
    content_format=ContentFormat.VIEW,
    page_ids=[],
    space_key="SEC",
    url="https://lftechnology.atlassian.net",
    username="prayatnakoirala@lftechnology.com",
)
faiss_index_path = "./rag_db"

embeddings = BedrockEmbeddings()
full_chunks = []

print("attempting to load from vector store")
try:
    vector_store = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
except Exception as e:
    vector_store = None
    print("vector store not found. creating a new one.")

if not vector_store:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_documents = []
    if pdf_file:
        print("adding PDF doument to vector store")
        all_documents.extend(pdf_file)
    if docx_file:
        print("adding DOCX doument to vector store")
        all_documents.extend(docx_file)
    try:
        confluence_docs = confluence.load()
        code_of_conduct_doc = []
        for doc in confluence_docs:
            if (
                doc.metadata.get("id") == "4071293249"
                and doc.metadata.get("title") == "Leapfrog Global Code of Conduct"
            ):
                code_of_conduct_doc.append(doc)

        if code_of_conduct_doc:
            print("adding confluence doument to vector store")
            all_documents.extend(confluence_docs)
    except Exception as e:
        print(f"warning: could not load Confluence docs: {e}")

    chunks = text_splitter.split_documents(all_documents)
    ids = list(range(len(chunks)))
    full_chunks.extend(list(zip(chunks, ids)))

    documents, ids = zip(*full_chunks)
    vector_store = FAISS.from_documents(documents, embeddings, ids=list(ids))
    FAISS.save_local(vector_store, faiss_index_path)
    print("vector store created!")

results = vector_store.similarity_search(query=user_prompt, k=8)
print(f"\nRetrieved from vector store: \n{results[0].page_content}")