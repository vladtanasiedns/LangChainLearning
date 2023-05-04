from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
import pinecone
import dotenv
import os

dotenv.load_dotenv()
print('Loading PDF...')
loader = UnstructuredPDFLoader('/home/build/auto_agent_skip/data/codul-muncii.pdf')
data = loader.load()
print(f'There are {len(data)} documents in the dataset')
print(f'There are {len(data[0].page_content)} pages in the first document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print(f'There are {len(texts)} texts in the dataset')

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)
indexn_name = 'cod-civil'
docsearch = Pinecone.from_texts(
    [t.page_content for t in texts],
    embedding=embeddings,
    index_name=indexn_name,
)

query = input('Cauta in codul muncii: ') 
template = """
    Esti un avocat expert in dreptul muncii.
    Furnizeaza toate informatiile legate de {query}
    Include toate explicatiile legate de subiect.
    Include articolele unde se pot gasi detalii.
    Inculde informatii din articolele conectate.
"""
prompt = PromptTemplate(
    input_variables=['query'],
    template=template,
)
prompt.format(query=query)
docs = docsearch.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
chain = load_qa_chain(llm, chain_type='stuff')

result = chain.run(input_documents=docs, question=prompt)
print(result)
