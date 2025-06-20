import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With OPENAI"

## promopt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant .Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

def genearte_response(question,api_key,llm,temperature,max_tokens):
    ##openai.api_key=api_key
    llm=ChatOpenAI(model=llm,
                   temperature=temperature,
                   max_tokens=max_tokens,
                   openai_api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## title of the app

st.title("Enhanced Q&A Chatbot with OpenAI")

## sidebar for settings

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Open AI API key",type="password")

## Drop down to select various Open AI models
llm=st.sidebar.selectbox("Select an Open AI Model", ["gpt-4o","gpt-4-turbo","gpt-4"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## main interface for user input
st.write("go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        response = genearte_response(user_input, api_key, llm, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide a query.")
