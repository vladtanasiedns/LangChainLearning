from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.text_splitter import TextSplitter
import pandas as pd
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_DATABASE = os.getenv('DB_DATABASE')

db = SQLDatabase.from_uri(f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_DATABASE}")
mysql = db._engine.raw_connection()
llm = ChatOpenAI(temperature=0, verbose=True)
#TODO: Provide the description of the tables in natural language

# Provide descriptions of the tables and the columns inside the tables
database_prompt = """
Given an input question or task, create a syntactically correct SQL query for a MySQL database. 
Then return the query but without the limit clause.

Only use the DBR table from the CDS database, this means prefixing the DBR table with `CDS` example `CDS.DBR`.
The most important table columns are: 
    column name: DBR_NO, desc: for the debtor number also the primary key, 
    column name: DBR_NAME1, desc: for the name, 
    column name: DBR_ASSIGN_AMT, desc: for the amount assigned as debt, 
    column name: DBR_CLIENT, desc: for the client number the debtor belongs to, example: DELL64, TUNG1, 
    column name: DBR_ASSIGN_DATE_O, desc: for the date on which the debtor was assigned,
there are other table columns but these are the only ones you need to know about.
Use only the column names above as they are named or use the `as` keyword to query them under another name.

Example query: `SELECT DBR_NAME, DBR_ASSIGN_AMT, DBR_ASSIGN_DATE_O FROM CDS.DBR WHERE DBR_CLIENT = 'DELL64'`

If someone asks for the table debtor, they really mean the DBR table from the CDS database, this means prefixing the DBR table with `CDS` example `CDS.DBR`.
Always limit the results to 10.
Question: {input}
"""

database_prompt_template = PromptTemplate(
    template=database_prompt,
    input_variables=["input"],
)

chain = SQLDatabaseChain.from_llm(db=db, prompt=database_prompt_template, verbose=True, llm=llm, top_k=3, )
query = chain.run(query="Give me a list of all debtors that have the client id DELL64, and an assign ammount between 1000 and 2000")

print(query, 'query')

pandas_prompt = """
You are a statistical analyst who is an expert at analyzing data.
Provide some general statistics on the data.
"""
df = pd.read_sql(query, mysql)


pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)
pandas_agent.run(pandas_prompt)
