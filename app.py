import google.generativeai as palm
import streamlit as st 
import os 

# Set your API key
palm.configure(api_key = os.environ['PALM_KEY'])

# Select the PaLM 2 model
model = 'models/text-bison-001'

# Generate text
if prompt := st.chat_input("Ask your query..."):
    enprom = f"""Answer the below provided input in context to Bhagwad Geeta. Use the verses and chapters sentences as references to your answer with suggestions
    coming from Bhagwad Geeta. Your answer to below input should only be in context to Bhagwad geeta only.\nInput= {prompt}"""
    completion = palm.generate_text(model=model, prompt=enprom, temperature=0.5, max_output_tokens=800)

# response = palm.chat(messages=["Hello."])
# print(response.last) #  'Hello! What can I help you with?'
# response.reply("Can you tell me a joke?")

# Print the generated text
    with st.chat_message("Assistant"):
        st.write(completion.result)








# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# # from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
# import os 
# # from transformers import T5Tokenizer, T5ForConditionalGeneration
# # from langchain.callbacks import get_openai_callback

# hub_token = os.environ["HUGGINGFACE_HUB_TOKEN"]

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=200,
#         chunk_overlap=20,
#         length_function=len
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     # embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#     embeddings = HuggingFaceEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     # llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
#     # tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
#     # model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

#     llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1", huggingfacehub_api_token=hub_token, model_kwargs={"temperature":0.5, "max_length":20})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation
#     reply = response.run(user_question)
#     st.write(reply)
#     # st.session_state.chat_history = response['chat_history']

#     # for i, message in enumerate(st.session_state.chat_history):
#     #     if i % 2 == 0:
#     #         st.write(user_template.replace(
#     #             "{{MSG}}", message.content), unsafe_allow_html=True)
#     #     else:
#     #         st.write(bot_template.replace(
#     #             "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             if(len(pdf_docs) == 0):
#                 st.error("Please upload at least one PDF")
#             else:
#                 with st.spinner("Processing"):
#                     # get pdf text
#                     raw_text = get_pdf_text(pdf_docs)

#                     # get the text chunks
#                     text_chunks = get_text_chunks(raw_text)

#                     # create vector store
#                     vectorstore = get_vectorstore(text_chunks)

#                     # create conversation chain
#                     st.session_state.conversation = get_conversation_chain(
#                         vectorstore)

# if __name__ == '__main__':
#     main()






# # import os
# # import getpass
# # import streamlit as st
# # from langchain.document_loaders import PyPDFLoader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import Chroma
# # from langchain import HuggingFaceHub
# # from langchain.chains import RetrievalQA
# # # __import__('pysqlite3')
# # # import sys
# # # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# # # load huggingface api key
# # hubtok = os.environ["HUGGINGFACE_HUB_TOKEN"]

# # # use streamlit file uploader to ask user for file
# # # file = st.file_uploader("Upload PDF")


# # path = "Geeta.pdf"
# # loader = PyPDFLoader(path)
# # pages = loader.load()

# # # st.write(pages)

# # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# # docs = splitter.split_documents(pages)

# # embeddings = HuggingFaceEmbeddings()
# # doc_search = Chroma.from_documents(docs, embeddings)

# # repo_id = "tiiuae/falcon-7b"
# # llm = HuggingFaceHub(repo_id=repo_id, huggingfacehub_api_token=hubtok, model_kwargs={'temperature': 0.2,'max_length': 1000})

# # from langchain.schema import retriever
# # retireval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=doc_search.as_retriever())

# # if query := st.chat_input("Enter a question: "):
# #   with st.chat_message("assistant"):
# #     st.write(retireval_chain.run(query))