import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain.messages import HumanMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vectorStore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:   
    {context}
    Question: {question}
    Provide a detailed answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query):
    """
    Simple retrieval chain without LCEL
    Manually retrieves documents, format them and generates response.

    Limitations:
    --Manual step by step execution
    --No built-in streaming support
    --No async support without additional code
    --Harder to compose with other chain
    --More verbose and error-prone
    """

    #########Retrieve relevant documents
    docs = retriever.invoke(query)

    #########Format documents into context string
    context = format_docs(docs)

    #########Format the prompt with context and question
    messages = prompt_template.format_prompt(question=query, context=context)

    response = llm.invoke(messages)

    return response.content


###############Implementation 2 : with LCEL : Better Approach########
def retrieval_chain_with_lcel():
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


def main():
    print("Retrieving...................")
    query = "What are 2 key learnings from Bhagwat Gita?"

    print("\n" + "#" * 50)
    print("Implementation 0: Raw LLM Invocation (No RAG)")
    print("#" * 50)

    raw_result = llm.invoke([HumanMessage(content=query)])
    print("Answer:")
    print(raw_result.content)

    print("\n" + "#" * 50)
    print("Implementation 1: Without LCEL")
    print("#" * 50)
    print("Answer:")
    result_without_lcel = retrieval_chain_without_lcel(query)
    print(result_without_lcel)

    print("\n" + "#" * 50)
    print("Implementation 2: With LCEL")
    print("#" * 50)
    print("Answer:")
    result_with_lcel = retrieval_chain_with_lcel().invoke({"question": query})
    print(result_with_lcel)


if __name__ == "__main__":
    main()
