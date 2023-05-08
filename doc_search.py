from langchain.document_loaders import UnstructuredPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import initialize_agent
from langchain import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
import dotenv
import os
from langchain.embeddings import OpenAIEmbeddings

dotenv.load_dotenv()

embeddings_size = 1536
index_name = 'cod-civil'
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

initial_prompt = """
    You are a seasoned romanian lawyer, expert in all legal matters.
    Respone to any following question with the correct article from the legal text.
    Provide explanations of the laws so that a child can understand them.
    
    The questions will be received in Romanian.
    The answers will be given in Romanian.
"""

memory = ConversationBufferMemory(memory_key='chat_history')

llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
chain = load_qa_chain(llm, chain_type='stuff', verbose=True)
# chain = initialize_agent(
#     llm=llm,
#     agent='conversational-react-description',
#     memory=memory,
#     verbose=True,
# )

# Forever loop
while True:
    print('Prompt: ')
    userInput = input()
    if userInput == 'exit':
        break
    docs = docsearch.similarity_search(userInput, top_k=10, include_metadata=True)
    chain.run(input_documents=docs, question=userInput)