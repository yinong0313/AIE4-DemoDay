from utils.prompts import AGENT_SYSTEM_PROMPT

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_openai import ChatOpenAI
from typing import List

def get_agent_prompt(prompt):
    system_prompt = prompt + AGENT_SYSTEM_PROMPT

    agent_prompt = ChatPromptTemplate.from_messages(
            [(
                    "system",
                    system_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
    return agent_prompt


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    
    agent_prompt = get_agent_prompt(prompt)
    agent = create_openai_functions_agent(llm, tools, agent_prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor
    

def create_team_supervisor(llm: ChatOpenAI, system_prompt, team_members) -> str:
    """An LLM-based router."""
    options = ["FINISH", "DataVis"] + team_members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                    },
                "question": {
                    "title": "Question",
                    "type": "string",
                    "description": "A clarifying question to ask if needed. Only required if 'next' is 'DataVis'.",
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, do we have an answer to the initial question with at least two references?"
                "If yes, let's FINISH."
                "Or should we ask a clarification question and visualise the data again, select 'DataVis'"
                "Or should we do some more document search? Select one of: {team_members}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(team_members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    
    # Function to truncate messages
def truncate_messages(messages: List[HumanMessage], max_tokens: int = 3000) -> List[HumanMessage]:
    """
    Truncate the messages to ensure the total token count stays within the limit.
    """
    total_tokens = sum(len(message.content.split()) for message in messages)  # Approximate token count
    while total_tokens > max_tokens and messages:
        messages.pop(0)  # Remove the oldest message to reduce token count
        total_tokens = sum(len(message.content.split()) for message in messages)
    return messages
