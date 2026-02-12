from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

textSplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250,chunk_overlap=0)
docsSplitted = textSplitter.split_documents(docs_list)

# vectorStore = Chroma.from_documents(
#     documents=docsSplitted,
#     collection_name="rags-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma"
# )

retriever = Chroma(
    collection_name="rags-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./.chroma"
).as_retriever()
