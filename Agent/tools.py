# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain.tools import Tool
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# import base64 
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv
# import os
# from langchain_core.tools import tool
# from langchain_community.tools.riza.command import ExecPython
# from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
# from langchain_community.vectorstores import FAISS
# import faiss
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain.tools.retriever import create_retriever_tool

# from agents_course_hf.agentic_rag import tools

# import os
# import gradio as gr
# import requests
# import inspect
# import pandas as pd
# import re
# import json
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph, END, START, MessagesState
# from langgraph.graph.message import add_messages
# from typing import TypedDict, Annotated, Optional
# from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
# from langgraph.prebuilt import ToolNode, tools_condition


# system_prompt = """
#         You are a general AI assistant. I will ask you a question.

#         First, explore your reasoning process step by step. Consider all relevant facts and possibilities.

#         Then, provide your answer using EXACTLY this format:

#         FINAL ANSWER: [ Your consice answer here]

#         Your FINAL ANSWER should be:
#         - For numbers: Just the number without commas or units (unless specified)
#         - For text: As few words as possible with no articles or abbreviations 
#         - For lists: Comma-separated values following the above rules

#         Important: The evaluation system will ONLY read what comes after "FINAL ANSWER:". Make sure your answer is correctly formatted.

#         """

# load_dotenv(r"C:\Projects\RAG_PoC\agents_course_hf\.env")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# vision_llm = ChatOpenAI(model="gpt-4o")
# search = DuckDuckGoSearchAPIWrapper()

# @tool
# def search_function(query: str) -> str:
#     """Search the web for information."""
#     try:
#         results = search.run(query)
#         if not results or results.strip() == "":
#             return "No search results found. Please try a different query."
#         return results
#     except Exception as e:
#         return f"Error during search: {str(e)}"


# @tool
# def image_describer(image_url: str) -> str:
#     """Describes the content of an image."""

#     description = ""

#     try:
#         import requests 
#         response = requests.get(image_url)
#         image_bytes = response.content
#         image_base64 = base64.b64encode(image_bytes).decode('utf-8')

#         message = [
#             HumanMessage(
#                 content=[
#                     {
#                     "type": "text",
#                     "text": (
#                         "Describe the type of image you see, if it is a photo, a drawing, a painting, etc. "
#                         "Then describe the content of the image in the most detailled way possible. "
#                         "You will start by describing the front of the image, then the back of the image if possible. "
#                         "If the image contains text, you will extract it and describe it in the most detailler way possible. "
#                         "If the image is a document, you will extract the text. Return only the text in this case, no explanations."
                        
#                         ),
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": f"data:image/png;base64,{image_base64}",
#                         }
#                     }
#                 ]
#             )
#         ]

#         # call the vision model
#         response = vision_llm(message)
#         description += response.content + "\n\n"

#         return description.strip()

#     except Exception as e:
#         print(f"Error reading image file: {e}")
#         return "Error reading image file."




# @tool
# def addition(a: int, b: int) -> int:
#     """Adds two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     return a + b


# @tool
# def subtract(a: int, b: int) -> int:
#     """Substract two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     return a - b

# @tool
# def multiply(a: int, b: int) -> int: 
#     """Multiply two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     return a * b

# @tool
# def divide(a: int, b: int) -> float:
#     """Divide two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     if b == 0:
#         return "Error: Division by zero is not allowed."
#     return a / b

# @tool 
# def modulus(a: int, b: int) -> int:
#     """Modulus two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     return a % b

# @tool
# def exponent(a: int, b: int) -> int:
#     """Exponent two numbers.
    
#     Args: 
#         a: first int
#         b: second int
#     """
#     return a ** b

# @tool
# def python_code_executor(code: str) -> str:
#     """ Executes a Python code snippet and returns the results
    
#     Args:
#         code: str, the Python code to execute
#     """
#     try:
#         exec_python = ExecPython()
#         result = exec_python.run(code)
#         return result
#     except Exception as e:
#         return f"Error executing code: {str(e)}"
    

# @tool 
# def wikipedia_search(query: str) -> str:
#     """Search Wikipedia for a given query and return the 2 first.
    
#     Args:
#         query: str, the search query
#     """

#     try:
#         search_documents = WikipediaLoader(query=query, max_results=2).load()
#         results = "\n".join([doc.page_content for doc in search_documents])
#         return results
#     except Exception as e:
#         return f"Error during Wikipedia search: {str(e)}"
    

# @tool
# def arvix_search(query: str) -> str:
#     """Search Arxiv for a query and return maximum 3 result.
    
#     Args:
#         query: The search query."""
#     search_docs = ArxivLoader(query=query, load_max_docs=3).load()
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
#             for doc in search_docs
#         ])
#     return {"arvix_results": formatted_search_docs}

# # retriever

# embeddings = OpenAIEmbeddings()

# index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={}
# )

# create_retriever_tool = create_retriever_tool(
#     retriever=vector_store.as_retriever(),
#     name="Question search", 
#     description="A tool to retrieve similar questions from a vector store."

# )

# tools = [
#     search_function,
#     image_describer,
#     addition,
#     subtract,
#     multiply,
#     divide,
#     modulus,
#     exponent,
#     python_code_executor,
#     wikipedia_search,
#     arvix_search,
#     create_retriever_tool
# ]


# def build_graph():
#     """Build the graph"""
#     chat = ChatOpenAI(model="gpt-4o")
#     chat_with_tools = chat.bind_tools(tools)

#     def assistant(state: MessagesState):
#         return {
#             "messages": [chat_with_tools.invoke(state["messages"])]
#         }
    
#     def retriever(state: MessagesState):
#         """Retriever node"""
#         similar_question = vector_store.similarity_search(state["messages"][0].content)
#         example_msg = HumanMessage(
#             content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
#         )
#         return {"messages": [system_prompt] + state["messages"] + [example_msg]}

#     builder = StateGraph(MessagesState)
#     builder.add_node("retriever", retriever)
#     builder.add_node("assistant", assistant)
#     builder.add_node("tools", ToolNode(tools))
#     builder.add_edge(START, "retriever")
#     builder.add_edge("retriever", "assistant")
#     builder.add_conditional_edges(
#         "assistant",
#         tools_condition,
#     )
#     builder.add_edge("tools", "assistant")

#     # Compile graph
#     return builder.compile()

# if __name__ == "__main__":
#     question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
#     # Build the graph
#     graph = build_graph()
#     # Run the graph
#     messages = [HumanMessage(content=question)]
#     messages = graph.invoke({"messages": messages})
#     for m in messages["messages"]:
#         m.pretty_print()