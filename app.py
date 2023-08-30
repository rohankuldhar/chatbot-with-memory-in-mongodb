import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from langchain.memory import MongoDBChatMessageHistory



def get_pdf_text(pdf_docs):
    text =''
    for pdf in pdf_docs:                # iterate through multiple pdfs
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:   # iterate through pages
            text+=page.extract_text()
    return text

def get_text_chunks(text):                      # use module character text splitter
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):        # openai embeddings to calculate embeddings and faiss is a vector store locally (to store on cloud use pinecone)
    embeddings=OpenAIEmbeddings()        # openai embeddings are paid , for free embeddings you can use instructor embeddings but it will run on local machine
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)    # to create memory
    conversation_chain = ConversationalRetrievalChain.from_llm(                        # to include context in chat
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    connection_string = "mongodb+srv://test:test@ronk.1ikvkur.mongodb.net/?retryWrites=true&w=majority"
    message_history = MongoDBChatMessageHistory(
    connection_string=connection_string, session_id = 'new_session')
    
    message_history.add_user_message(response["question"])
    message_history.add_ai_message(response["answer"])

    for i,message in enumerate(st.session_state.chat_history):
        if i %2==0:
            st.write(user_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
  
def main():
    load_dotenv()
    

    st.set_page_config(page_title='Chat with PDF',page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation=None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=None

    st.header('Chat with PDF :books:')
    user_question = st.text_input('Ask question about documents')
    if user_question:
        handle_userinput(user_question)

    # to put things inside sidebar , use 'with' and dont add parenthesis at the end of function

    with st.sidebar:
        st.subheader('Your documents')
        # save uploaded pdf in variable pdf_docs
        pdf_docs=st.file_uploader('Upload your PDF here',accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing'):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks= get_text_chunks(raw_text)
                

                # create vector store
                vectorstore=get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)    # it takes the history of conversation and returns next
                



if __name__ == '__main__':
    main()
