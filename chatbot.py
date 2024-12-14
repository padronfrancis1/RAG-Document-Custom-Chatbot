from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided, answer
with "I don't know".
Context: Quantum computing is an emerging field that leverages quantum mechanics to solve complex problems faster than classical computers.
...
Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt_template)

input_data = {"query": """What is the main advantage of quantum computing over classical computing?"""}
 
response = chain.run(input_data)
print("Question:", input_data["query"])
print("Answer:", response)
