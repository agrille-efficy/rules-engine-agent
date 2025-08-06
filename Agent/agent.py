import base64 
import os
import requests
import uuid
import faiss
import tempfile

import pandas as pd
import numpy as np

from typing import Optional, Dict, Any, List 
from dotenv import load_dotenv
from urllib.parse import urlparse 
from image_processing import *
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from code_interpreter import CodeInterpreter
from openai import OpenAI

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, MessagesState

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

# Import the focused tools for file processing with PDF support
from tools import tools as file_processing_tools


# Load environment variables - works both locally and on Hugging Face spaces
load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Vision model for image processing
vision_llm = ChatOpenAI(temperature=0) 


@tool
def image_describer(image_url: str) -> str:
    """Describes the content of an image."""

    description = ""

    try:
        import requests 
        response = requests.get(image_url)
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        message = [
            HumanMessage(
                content=[
                    {
                    "type": "text",
                    "text": (
                        "Describe the type of image you see, if it is a photo, a drawing, a painting, etc. "
                        "Then describe the content of the image in the most detailled way possible. "
                        "You will start by describing the front of the image, then the back of the image if possible. "
                        "If the image contains text, you will extract it and describe it in the most detailler way possible. "
                        "If the image is a document, you will extract the text. Return only the text in this case, no explanations."
                        
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        }
                    }
                ]
            )
        ]

        # call the vision model
        response = vision_llm.invoke(message)
        description += response.content + "\n\n"

        return description.strip()

    except Exception as e:
        print(f"Error reading image file: {e}")
        return "Error reading image file."





@tool 
def code_executor(code: str, language : str = "python") -> str: 
    """
    Executes a code snippet and returns the results.

    Supports python, bash, sql, c, java

    Args: 
        code: str, the code to execute
        language: str, the programming language of the code snippet (python by default)

    Returns: 
        str: the result of the code execution or an error message if execution fails.
    """
    try:
        interpreter = CodeInterpreter()
        result = interpreter.execute_code(code, language=language)
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"



@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."


@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"

@tool
def extract_structured_data_from_image(image_path: str) -> str:
    """
    Classify the document and extract structured data from an image using a vision model. 
    Args:
        image_base64 (str): Base64 encoded image string
    Returns: 
        str: Vision model response with structured data extraction.
    """
    try:

        loaded_img = load_local_image(image_path)
        base64_image = encode_image(loaded_img)

        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.response.create(
            model=vision_llm.model.name,

            input=[ 
            {
                "role" : "user",
                "content" : [ 
                    {"type": "input_text", "text": "Read the following document, identify which kind of document it is and organize the data into structured data."},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}",
                    }
                ],
            },
            ],
        )
        return response.output_text
    
    except Exception as e:
        return f"Error extracting structured data from image: {str(e)}" 



@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"


@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"


# Load system prompt from file
try:
    with open("Agent\system_prompt.txt", "r") as f:
        system_prompt = f.read()
except FileNotFoundError:
    raise FileNotFoundError("system_prompt.txt file is required. Please ensure it exists in the current directory.")

sys_msg = SystemMessage(content=system_prompt)

# retriever
embeddings = OpenAIEmbeddings()

# Load vector store
try:
    
    vector_store_paths = [
        "vector_store",  # Current directory
        "./vector_store",  # Relative path
        r"C:\Projects\RAG_PoC\agents_course_hf\Final_Assignment_Template\vector_store"  # Original local path
    ]
    
    vector_store = None
    for path in vector_store_paths:
        try:
            vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            print(f"Vector store loaded from: {path}")
            break
        except:
            continue
    
    if vector_store is None:
        print("Warning: Vector store not found. Creating empty vector store.")
        # Create a minimal vector store with dummy data
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(embeddings, index, {}, {})

except Exception as e:
    print(f"Warning: Could not load vector store: {e}")
    # Create a minimal vector store with dummy data
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(embeddings, index, {}, {})

create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question_search", 
    description="A tool to retrieve similar questions from a vector store."
)

# Focused tools for file processing, vision analysis, and database ingestion
tools = file_processing_tools + [create_retriever_tool]


def build_graph():
    """Build the graph"""
    chat = ChatOpenAI(model="gpt-4o")
    chat_with_tools = chat.bind_tools(tools)

    def assistant(state: MessagesState):
        return {
            "messages": [chat_with_tools.invoke(state["messages"])]
        }
    
    def retriever(state: MessagesState):
        """Retriever node"""
        try:
            similar_question = vector_store.similarity_search(
            query=state["messages"][0].content,
            k=1
            )
            if not similar_question:
                example_msg = HumanMessage(
                    content="No similar questions were found in the vector store."
                )
                print("No similar questions found")
            else:
                example_msg = HumanMessage(
                    content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
                )
                print("Similar question found for reference")
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        except Exception as e:
            print(f"Retriever error: {e}")
            # Return minimal state if retriever fails
            return {"messages": [sys_msg] + state["messages"]}

    print("Setting up graph nodes...")
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    print("Graph compilation complete")
    # Compile graph
    return builder.compile()

if __name__ == "__main__":
    question = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? " \
    "You can use the latest 2022 version of english wikipedia."

    # Build the graph
    graph = build_graph()
    # Run the graph
    
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()


## TODO: ADD MEMORY TO THE AGENT

