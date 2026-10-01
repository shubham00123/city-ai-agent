
from dotenv import load_dotenv

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os


search_tool = TavilySearch(max_results=5)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


prompt = ChatPromptTemplate.from_template(
    """
you are a helpful assistant

summarize the following news into clear bullet points

{news}
"""
)


chain = prompt | llm | StrOutputParser()

news_result = search_tool.invoke({"query": "latest news about AI"})

result = chain.invoke({"news": news_result})

print(result)
