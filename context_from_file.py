from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

with open('./context.txt', 'r', encoding='utf-8') as file:
    context = file.read()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """
Answer the question based on the context below. 
If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {query}
Answer: """

prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt_template)

input_data = {
    "context": context,
    "query": "Please give me a summary of the abstract section os this document."
}

response = chain.run(input_data)
print("Question:", input_data["query"])
print("Answer:", response)
