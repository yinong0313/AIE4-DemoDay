from langchain_experimental.openai_assistant import OpenAIAssistantRunnable
from langchain_core.messages import HumanMessage
from openai import OpenAI
from utils.prompts import DATA_ANALYSIS_PROMPT


def create_thread(file_path, content):    
    """ Define the thread that uploads file and takes input message"""
    
    client = OpenAI()
    file = client.files.create(
                file=open(file_path, "rb"),
                purpose='assistants'
                )

    thread = client.beta.threads.create(
                    messages=[{"role": "user",
                                "content": content,
                                "attachments": [{
                                    "file_id": file.id,
                                    "tools": [{"type": "code_interpreter"}]
                                    }]
                            }],           
                    )
    return client, thread

    

def data_visualization_node(content, file_path, prompt=DATA_ANALYSIS_PROMPT, name='data_vis'):
    assistant = OpenAIAssistantRunnable.create_assistant(
                            name="visualization_assistant",
                            instructions = prompt,
                            tools=[{"type": "code_interpreter"}],
                            model="gpt-3.5-turbo", 
                            truncation_strategy={
                                        "type": "last_messages",
                                        "last_messages": 1
                                    })
    client, thread = create_thread(file_path, content)
    results = assistant.invoke(input={"content": content, 'thread_id':thread.id})[-1]
    
    # retrieve image
    f_id = results.content[0].image_file.file_id
    image_data = client.files.content(f_id)
    image_data_bytes = image_data.read()
    with open("data/plot.png", "wb") as file:
        file.write(image_data_bytes)
    
    return {"messages": [HumanMessage(content=results.content[1].text.value, name=name)]}
