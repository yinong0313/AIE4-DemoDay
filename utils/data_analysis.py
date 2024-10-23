from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langchain_core.messages import HumanMessage
from openai import OpenAI

from utils.prompts import DATA_ANALYSIS_PROMPT, DATA_ANALYSIS_INSTRUCTION
from utils.models import REASONING_LLM_ID
import functools
from datetime import datetime

# def create_thread(state):    
#     """ Define the thread that uploads file and takes input message"""
    
#     client = OpenAI()
#     file = client.files.create(
#                 file=open(state['file_path'], 'rb'),
#                 purpose='assistants'
#                 )

#     thread = client.beta.threads.create(
#                     messages=[{"role": "user",
#                                 "content": state['questions'][-1].content,
#                                 "attachments": [{
#                                     "file_id": file.id,
#                                     "tools": [{"type": "code_interpreter"}]
#                                     }]
#                             }],           
#                     )
#     return client, thread

    

# def assistant_node(state, assistant, name):
    
#     # global client, thread_id
#     client, thread = create_thread(state)
#     thread_id = thread.id
#     results = assistant.invoke(input={"content": state['questions'][-1].content, 'thread_id':thread_id})[-1]
#     try:
#         # retrieve image
#         f_id = results.content[0].image_file.file_id
#         image_data = client.files.content(f_id)
#         image_data_bytes = image_data.read()
        
#         time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         with open(f"data/data_visualisation/plot_{time_stamp}.png", "wb") as file:
#             file.write(image_data_bytes)
        
#         return {"messages": [HumanMessage(content=results.content[1].text.value, name=name)]}
#     except:
#         return {"messages": [HumanMessage(content=results.content[0].text.value, name=name)]}

# def data_visualization_node(prompt=DATA_ANALYSIS_PROMPT, name='DataVis'):
#     assistant = OpenAIAssistantRunnable.create_assistant(
#                             name="visualization_assistant",
#                             instructions = prompt,
#                             tools=[{"type": "code_interpreter"}],
#                             model=REASONING_LLM_ID, 
#                             truncation_strategy={
#                                         "type": "last_messages",
#                                         "last_messages": 1
#                                     })
#     return functools.partial(assistant_node, assistant=assistant, name=name)

import logging


logging.basicConfig(level=logging.INFO)

def create_thread(state):    
    """ Define the thread that uploads file and takes input message """
    
    client = OpenAI()
    try:
        # Open the file and upload it
        with open(state['file_path'], 'rb') as f:
            file = client.files.create(
                file=f,
                purpose='assistants'
            )
        
        if not file.id:
            logging.error("File upload failed. No file ID returned.")
            return None, None
        
        logging.info(f"File uploaded successfully with ID: {file.id}")

        # Create the thread with the uploaded file
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": state['questions'][-1].content,
                "attachments": [{
                    "file_id": file.id,
                    "tools": [{"type": "code_interpreter"}]
                }]
            }]
        )
        
        logging.info(f"Thread created successfully with ID: {thread.id}")
        return client, thread

    except Exception as e:
        logging.error(f"Error while creating thread: {e}")
        return None, None

def assistant_node(state, assistant, name):
    # Create thread and get client
    client, thread = create_thread(state)
    if not client or not thread:
        return {"messages": [HumanMessage(content="Failed to create thread or upload file.", name=name)]}
    
    thread_id = thread.id
    try:
        
        input_content = DATA_ANALYSIS_PROMPT + state['questions'][-1].content
        
        # Invoke the assistant with input data
        results = assistant.invoke(input={"content": input_content, 'thread_id': thread_id})[-1]
        
        # Debug the results structure
        logging.info(f"Assistant invocation results: {results}")

        # Retrieve image if available
        if hasattr(results.content[0], 'image_file') and results.content[0].image_file:
            f_id = results.content[0].image_file.file_id
            image_data = client.files.content(f_id)
            image_data_bytes = image_data.read()
            
            # Save the image data to file
            time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = f"data/data_visualisation/plot_{time_stamp}.png"
            with open(file_path, "wb") as file:
                file.write(image_data_bytes)
            
            logging.info(f"Image saved successfully to: {file_path}")
            return {"messages": [HumanMessage(content=results.content[1].text.value, name=name, image_path=file_path)]}
        else:
            logging.warning("No image file found in the assistant's response.")
            return {"messages": [HumanMessage(content=results.content[0].text.value, name=name)]}

    except Exception as e:
        logging.error(f"Error while invoking assistant or processing results: {e}")
        return {"messages": [HumanMessage(content="An error occurred during the assistant invocation.", name=name)]}

def data_visualization_node(instructions=DATA_ANALYSIS_INSTRUCTION, name='DataVis'):
    # Create an assistant using OpenAIAssistantRunnable
    assistant = OpenAIAssistantRunnable.create_assistant(
        name="visualization_assistant",
        instructions=instructions,
        tools=[{"type": "code_interpreter"}],
        model=REASONING_LLM_ID,
        truncation_strategy={
            "type": "last_messages",
            "last_messages": 1
        }
    )
    # Return a function (using functools.partial) that can be called later with state as an argument
    return functools.partial(assistant_node, assistant=assistant, name=name)