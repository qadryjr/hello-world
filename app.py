
import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from .htmlTemplates import css, bot_template, user_template
from PIL import Image

PDF_DIRECTORY = "pdf_files"  # Specify the directory for PDF files

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_number].extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama2-uncensored:7b-chat")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def process_pdf():
    pdf_docs = [os.path.join(PDF_DIRECTORY, pdf) for pdf in os.listdir(PDF_DIRECTORY)]

    if "vectorstore" not in st.session_state:
        raw_text = get_pdf_text(pdf_docs)
        print("Extracted Text from PDFs:", raw_text)
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.vectorstore = vectorstore
        print("Vectorstore Contents:", vectorstore)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        print("Conversation Chain:", st.session_state.conversation)
    else:
        vectorstore = st.session_state.vectorstore

    #st.success("PDFs processed successfully")

chat_bubble_css = """
<style>
.user-message {
    margin: 5px;
    padding: 10px;
    background-color: #007BFF;
    color: white;
    border-radius: 15px;
    max-width: 60%;
    word-wrap: break-word;
    float: right;
    clear: both;
}

.bot-message {
    margin: 5px;
    padding: 10px;
    background-color: #E0E0E0;
    border-radius: 15px;
    max-width: 60%;
    word-wrap: break-word;
    float: left;
    clear: both;
}
</style>
"""

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        if st.session_state.get("chat_history"):
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User message
                    st.markdown(f'<div class="user-message">{message.content}</div>', unsafe_allow_html=True)
                else:  # Bot message
                    st.markdown(f'<div class="bot-message">{message.content}</div>', unsafe_allow_html=True)

    else:
        st.warning("Conversation not initialized. Please process PDFs first.")

def main():
    load_dotenv()
   # st.set_page_config("HR in PDFs Chatbot", page_icon=":scroll:")
    st.markdown(chat_bubble_css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Initialize conversation when the app starts
    process_pdf()

    # Display typing message only if it's the first time or the page is refreshed
    if "typing_message_displayed" not in st.session_state:
        st.session_state.typing_message_displayed = True
    else:
        st.session_state.typing_message_displayed = False

    st.header(" 🙌 HR Bot")
    user_question = st.text_input("Ask any Question related to our Vodafone HR section .. ✍️📝")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        #st.image("../images/vodafone_logo.png" ,width=90)
        image = Image.open('../images/vodafone_logo.png')
        st.write("---")
        #st.title("📁 Vodafone PDF  HR Section")
        col1, col2, col3 = st.columns(3)
        with col1:
             st.write(' ')
        with col2:
             st.image(image,width=90)
             st.write(' ')
            #st.write(' ')
        with col3:
            st.write(' ')

        st.markdown("## Welcome HR Bot")
        st.markdown("---")

        st.markdown(
            "## Meet the HR Bot! \n"
            "An HR chatbot is a virtual assistant representing a company's HR department. \nIt harnesses the power of AI to converse with employees and automate HR operations like \nrecruitment, onboarding, driving and enhancing the overall employee experience digitally."
        )

        st.markdown("---")

        st.markdown(
            "## How to use\n"
            "1. Ask HR question\n"
            "2. Wait for the analysis to complete\n"
            "3. Enjoy the feedback 🤗\n"
        )

        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY)

        # Typing message with animation (conditionally displayed)
        #if st.session_state.typing_message_displayed:
         #   st.markdown(
          #      """
           #     <style>
            #    @keyframes typeText {
             #       from { width: 0; }
              #      to { width: 100%; }
               # }

                #.typing-text {
                 #   white-space: nowrap;
                  #  overflow: hidden;
                   # animation: typeText 4s steps(50, end) infinite;
                #}
                #</style>
                #<div class="typing-text">
                #    <b>Processing PDFs... You can start chatting now.</b>
                #</div>
                #""",
                #unsafe_allow_html=True
            #)

    st.write("---")
    #if st.session_state.get("process_button_clicked", False):
       # st.image("img/Robot.jpg", width=150)  # Adjust the width as needed
        #st.write("App created by @ Vodafone.com")
    #else:
        #st.image("img/Robot.jpg", width=150)  # Adjust the width as needed
       # st.write("App created by @ Vodafone.com")

    st.markdown(
        """
        <style>
        @keyframes moveText {
            0% { transform: translateX(0); }
            25% { transform: translateX(5px); }
            50% { transform: translateX(0); }
            75% { transform: translateX(-5px); }
            100% { transform: translateX(0); }
        }

        .moving-text {
            animation: moveText 2s infinite;
        }
        </style>
        <div class="moving-text" style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #e60000; padding: 10px; text-align: center; color: #ffffff;">
            © <a href="https://vodafone.com" target="_blank" style="color: #ffffff;">Vodafone</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
