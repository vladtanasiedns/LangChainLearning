import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY 

from collections import deque
from typing import Dict, List, Optional, Any
from langchain import LLMChain

if __name__ == "__main__":
    print("Hello World!")