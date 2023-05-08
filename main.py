import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 

from typing import Optional
from langchain import LLMChain, OpenAI, PromptTemplate, SerpAPIWrapper 
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import BabyAGI
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
import faiss

# Task creation, prioritization and chain to execute
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor


embedings_model = OpenAIEmbeddings()
embeding_size = 1536
index = faiss.IndexFlatL2(embeding_size)
vectorstore = FAISS(embedings_model.embed_query, index, InMemoryDocstore({}), {})

todo_prompt = PromptTemplate.from_template(
    "Yout are a planner who is an expert at commping up with a todo list for a given objective. Come up with a todo list from this objective: {objective}",
)
todo_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=todo_prompt,
)
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful to search the web for information about a certain company or person in order to find contact details, addresses and other",
    ),
    Tool(
        name="TODO",
        func=todo_chain.run,
        description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
    )
]

prefix = """
You are an AI who performs one task based on the following objective: {objective}. 
Take into account these previously completed tasks: {context}. 
"""
suffix = """
Question: {task}
{agent_scratchpad}
"""

prompt = ZeroShotAgent.create_prompt(
    tools=tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["objective", "task", "context", "agent_scratchpad"],
)

llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
)

tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    allowed_tools=tool_names,
)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

max_iterations: Optional[int] = 3
baby_agi = BabyAGI.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    task_execution_chain=agent_executor,
    verbose=False,
    max_iterations=max_iterations,
)

OBJECTIVE = """
    Find any information about Drybar Holdings LLC located in Irvine, CA, find any relevant websites, contact details, addresses and other information. 
    Provide information on possible employees, their contact details and other relevant information.
    Include any relevant links found in the search.
    """
if __name__ == "__main__":  
    baby_agi({"objective": OBJECTIVE})