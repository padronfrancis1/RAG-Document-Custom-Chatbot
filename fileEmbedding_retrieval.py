from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("./document_sample.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
faiss_index = FAISS.from_documents(chunks, embeddings)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
retriever = faiss_index.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What is the main point of the document?"
response = qa_chain.run(query)
print("Question:", query)
print("Answer:", response)
