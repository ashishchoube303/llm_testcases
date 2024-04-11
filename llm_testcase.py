
from langchain_openai import OpenAI

# We are importing secreat key from secretkey.py file which is stored in our folder to make confidentiality
# Added
from constant import secreat_key

llm_prompt=OpenAI(openai_api_key=secreat_key)

from langchain.prompts import PromptTemplate

# prompt_template = PromptTemplate(
#     input_variables=["product"],
#     template="Given a {product} with a single login button. Write test scenarios about the {product}?",
# )

# response=llm_prompt.invoke("Web application")
# print(response)

from langchain.prompts.few_shot import FewShotPromptTemplate

examples = [
    {"Usecase": "Given a web application with a single login button. Write test scenarios about the login button in the web application?",
    "Test cases": 
"""
1. User logs in successfully.
2. User enters incorrect login credentials and receives an error message.
3. User enters login credentials for a non-existent account and receives an error message.
4. User enters login credentials for an account that has been deactivated or disabled.
5. User enters login credentials for an account that has been locked.
"""},
    
    {"Usecase": "Given a REST API which accepts a POST request. Write test scenarios about the API?",
    "Test cases": 
"""
1. User sends valid request to the API and recieves valid response.
2. User sends an empty body to the API and recieves an error message.
3. User sends a request with invalid fields to the API and recieves an error mesage.
"""},
    
    {"Usecase": "Given a ETL pipeline which extracts, transforms and loads data from a source table to a target table. Write test scenarios about the data ETL pipeline?",
    "Test cases": 
"""
1. Verify that the data in the target database is the same as the data in the source.
2. Verify that the constraints in the target database is as per the functional specification.
3. Verify that the target database does not has any empty field.
4. Verify that the target database does not has any duplicate values.
"""}
           ]


example_prompt = PromptTemplate(input_variables=["Usecase", "Test cases"], template="Usecase: {Usecase}\n{Test cases}")

few_shot_prompt = FewShotPromptTemplate(
    examples=examples, 
    example_prompt=example_prompt, 
    suffix="Usecase: {input}", 
    input_variables=["input"]
)


# response=llm_prompt.invoke(myprompt)
# print(response)

import streamlit as st

st.title("Testcase Generation with Help of LLM and Langchain")

input_text=st.text_input("Search the usecases which we required")

myprompt=few_shot_prompt.format(input=input_text)

if input_text:
    st.write(llm_prompt.invoke(myprompt))