from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import base64 
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_community.tools.riza.command import ExecPython
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.vectorstores import FAISS
import faiss
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, MessagesState
from typing import Optional, Dict, Any, List 
import numpy as np
from image_processing import *
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
import pandas as pd
import tempfile
import requests 
import uuid
from urllib.parse import urlparse 
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import pytesseract


load_dotenv(r"C:\Projects\RAG_PoC\agents_course_hf\.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

vision_llm = ChatOpenAI(temperature=0)

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return summarized results."""
    try:
        # Perform the search using TavilySearchResults
        search_docs = TavilySearchResults(max_results=3).invoke(query)
        
        # Format the results into a readable string
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                f"Page: {doc.metadata.get('page', 'N/A')}\n"
                f"Content: {doc.page_content[:500]}..."  # Limit content to 500 characters
                for doc in search_docs
            ]
        )
        
        return formatted_search_docs
    except Exception as e:
        return f"Error during web search: {str(e)}"


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
def add(a: int, b: int) -> int:
    """Adds two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Substract two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int: 
    """Multiply two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    return a * b

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a / b

@tool 
def modulus(a: int, b: int) -> int:
    """Modulus two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    return a % b

@tool
def exponent(a: int, b: int) -> int:
    """Exponent two numbers.
    
    Args: 
        a: first int
        b: second int
    """
    return a ** b

@tool
def python_code_executor(code: str) -> str:
    """ Executes a Python code snippet and returns the results
    
    Args:
        code: str, the Python code to execute
    """
    try:
        exec_python = ExecPython()
        result = exec_python.run(code)
        return result
    except Exception as e:
        return f"Error executing code: {str(e)}"
    

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return summarized results."""
    try:
        search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
        summarized_results = []
        for doc in search_docs:
            content = doc.page_content
            # Summarize or extract key sections
            summarized_results.append(content[:500])  # First 500 characters as a fallback

        return "\n\n---\n\n".join(summarized_results)
    except Exception as e:
        return f"Error during Wikipedia search: {str(e)}"
    

@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}

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
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    try:
        # Open the image
        image = Image.open(image_path)

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


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
    
@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}

        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)

        return {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def transform_image(
    image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            img = img.resize(
                (
                    params.get("width", img.width // 2),
                    params.get("height", img.height // 2),
                )
            )
        elif operation == "rotate":
            img = img.rotate(params.get("angle", 90), expand=True)
        elif operation == "crop":
            img = img.crop(
                (
                    params.get("left", 0),
                    params.get("top", 0),
                    params.get("right", img.width),
                    params.get("bottom", img.height),
                )
            )
        elif operation == "flip":
            if params.get("direction", "horizontal") == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif operation == "adjust_brightness":
            img = ImageEnhance.Brightness(img).enhance(params.get("factor", 1.5))
        elif operation == "adjust_contrast":
            img = ImageEnhance.Contrast(img).enhance(params.get("factor", 1.5))
        elif operation == "blur":
            img = img.filter(ImageFilter.GaussianBlur(params.get("radius", 2)))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif operation == "grayscale":
            img = img.convert("L")
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"transformed_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def draw_on_image(
    image_base64: str, drawing_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Draw shapes (rectangle, circle, line) or text onto an image.
    Args:
        image_base64 (str): Base64 encoded input image
        drawing_type (str): Drawing type
        params (Dict[str, Any]): Drawing parameters
    Returns:
        Dictionary with result image (base64)
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        color = params.get("color", "red")

        if drawing_type == "rectangle":
            draw.rectangle(
                [params["left"], params["top"], params["right"], params["bottom"]],
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "circle":
            x, y, r = params["x"], params["y"], params["radius"]
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "line":
            draw.line(
                (
                    params["start_x"],
                    params["start_y"],
                    params["end_x"],
                    params["end_y"],
                ),
                fill=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "text":
            font_size = params.get("font_size", 20)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(
                (params["x"], params["y"]),
                params.get("text", "Text"),
                fill=color,
                font=font,
            )
        else:
            return {"error": f"Unknown drawing type: {drawing_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"result_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def generate_simple_image(
    image_type: str,
    width: int = 500,
    height: int = 500,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a simple image (gradient, noise, pattern, chart).
    Args:
        image_type (str): Type of image
        width (int), height (int)
        params (Dict[str, Any], optional): Specific parameters
    Returns:
        Dictionary with generated image (base64)
    """
    try:
        params = params or {}

        if image_type == "gradient":
            direction = params.get("direction", "horizontal")
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))

            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            if direction == "horizontal":
                for x in range(width):
                    r = int(
                        start_color[0] + (end_color[0] - start_color[0]) * x / width
                    )
                    g = int(
                        start_color[1] + (end_color[1] - start_color[1]) * x / width
                    )
                    b = int(
                        start_color[2] + (end_color[2] - start_color[2]) * x / width
                    )
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            else:
                for y in range(height):
                    r = int(
                        start_color[0] + (end_color[0] - start_color[0]) * y / height
                    )
                    g = int(
                        start_color[1] + (end_color[1] - start_color[1]) * y / height
                    )
                    b = int(
                        start_color[2] + (end_color[2] - start_color[2]) * y / height
                    )
                    draw.line([(0, y), (width, y)], fill=(r, g, b))

        elif image_type == "noise":
            noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")

        else:
            return {"error": f"Unsupported image_type {image_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"generated_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def combine_images(
    images_base64: List[str], operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Combine multiple images (collage, stack, blend).
    Args:
        images_base64 (List[str]): List of base64 images
        operation (str): Combination type
        params (Dict[str, Any], optional)
    Returns:
        Dictionary with combined image (base64)
    """
    try:
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}

        if operation == "stack":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                new_img = Image.new("RGB", (total_width, max_height))
                x = 0
                for img in images:
                    new_img.paste(img, (x, 0))
                    x += img.width
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                new_img = Image.new("RGB", (max_width, total_height))
                y = 0
                for img in images:
                    new_img.paste(img, (0, y))
                    y += img.height
        else:
            return {"error": f"Unsupported combination operation {operation}"}

        result_path = save_image(new_img)
        result_base64 = encode_image(result_path)
        return {"combined_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


with open("agents_course_hf\Final_Assignment_Template\system_prompt.txt", "r") as f:
    system_prompt = f.read()

sys_msg = SystemMessage(content=system_prompt)

# retriever

embeddings = OpenAIEmbeddings()

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS.load_local(r"C:\Projects\RAG_PoC\agents_course_hf\Final_Assignment_Template\vector_store", embeddings, allow_dangerous_deserialization=True)


create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question_search", 
    description="A tool to retrieve similar questions from a vector store."

)

tools = [
    # Web and Knowledge Tools
    web_search,
    wiki_search,
    arxiv_search,
    
    # Image Processing Tools
    image_describer,
    extract_text_from_image,
    analyze_image,
    transform_image,
    draw_on_image,
    generate_simple_image,
    combine_images,
    
    # Math Operations
    add,
    subtract,
    multiply,
    divide,
    modulus,
    exponent,
    
    # File Handling Tools
    save_and_read_file,
    download_file_from_url,
    analyze_csv_file,
    analyze_excel_file,
    
    # Code Execution
    python_code_executor
]


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
        similar_question = vector_store.similarity_search(
            query=state["messages"][0].content,
            k=1
        )
        if not similar_question:
            example_msg = HumanMessage(
                content="No similar questions were found in the vector store."
            )
        else:
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
            )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

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