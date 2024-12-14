from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("./document_sample.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_documents(chunks, embeddings)

prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Please provide a detailed answer, citing specific information from the context when possible.
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = faiss_index.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 relevant chunks
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

query = "What is the main point of the document?"
response = qa_chain({"query": query})
print("Question:", query)
print("Answer:", response["result"])
print("\nSource Documents:")
for doc in response["source_documents"]:
    print("\n---")
    print(doc.page_content[:200] + "...")