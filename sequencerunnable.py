import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Prompt Template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)


# 2. Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)


# 3. Output Parser
parser = StrOutputParser()


chain = prompt | model | parser


result = chain.invoke("Machine Learning")
print(result)
