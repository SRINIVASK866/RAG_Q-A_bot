#importing all the necessary packages
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

#Loading the file
pdf = PyPDFLoader("C:\\Users\\srinivas\\Desktop\\Innomatics Intership\\GenAI pdf\\No context behind paper.pdf")

#Initializing the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key,
                                               model="models/embedding-001")

#storing and retriving using chromadb
db_connection=Chroma(persist_directory="./chroma_db_", embedding_function= embedding_model)

retriver = db_connection.as_retriever(search_kwargs={"k":5})

#creating chat template
chat_template = ChatPromptTemplate.from_messages([
    #System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Bot.
    You take the content and question from user. Your answer should be based on the specific content."""),
    #Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}

    Question:
    {question}
    Answer:""")
])

#Initializing chat model
chat_model = ChatGoogleGenerativeAI(google_api_key= "AIzaSyAZPv1J2lWwq_mTqaTlB5qQo4Iiy-IviPk",
                                    model="gemini-1.5-pro-latest")


#output parsing
output_parser = StrOutputParser()

#Initializing RAG chain
def format_docs(docs):
  return "\n\n".join(doc.pagecontent for doc in docs)

rag_chain = (
    {"context":retriver | format_docs, "question":RunnablePassthrough()}
    |chat_template
    |chat_model
    |output_parser
)

st.title("RAG system")
st.subheader("A RAG System that answers question related to “Leave No Context Behind” Paper")



query = st.text_input("Enter your question")

if query:
  response = rag_chain.invoke(query)
  st.write(response)

st.button("Answer Me")
