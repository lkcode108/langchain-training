import os

from dotenv import load_dotenv
from langchain_classic.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()
print("Loading.............")
loader = TextLoader(
    "C:\\Users\\lsingh38\\PycharmProjects\\langchain-training\\BhagwatGita.txt",
    encoding="latin-1",
)
document = loader.load()

print("Splitting.................")
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print(f"number of text chunks: {len(texts)}")

print("Embedding.................")
embeddings = OpenAIEmbeddings()

print("Deleting previous records from pinecone index")
PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"]).delete(
    namespace="__default__", delete_all=True
)

print("Ingesting.............")

PineconeVectorStore.from_documents(
    texts, embeddings, index_name=os.environ["INDEX_NAME"]
)

print("Finish")


def main():
    print("Hello from langchain-training!")


if __name__ == "__main__":
    main()
