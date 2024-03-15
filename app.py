import os, dotenv, anthropic, panel, openai
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

panel.extension()

# Set API key
dotenv.load_dotenv()
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

@panel.cache
def load_vectorstore():
    # If the vector embeddings of the documents have not been created
    if not os.path.isfile('chroma_db/chroma.sqlite3'):

        # Load the documents
        loader = DirectoryLoader('Docs/', glob="./*.pdf", loader_cls=PyPDFLoader)
        data = loader.load()

        # Split the docs into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )
        docs = splitter.split_documents(data)

        # Embed the documents and store them in a Chroma DB
        embedding=OpenAIEmbeddings(openai_api_key = openai.api_key)
        vectorstore = Chroma.from_documents(documents=docs,embedding=embedding, persist_directory="./chroma_db")
    else:
        # load ChromaDB from disk
        embedding=OpenAIEmbeddings(openai_api_key = openai.api_key)
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

    return vectorstore

# Initialize the chat history
chat_history = []

async def get_response(contents, user, instance):

    # Load the vectorstore
    vectorstore = load_vectorstore()

    question = contents

    # Get the relevant information to form the context on which to query with the LLM
    docs = vectorstore.similarity_search(question)

    context = "\n"
    for doc in docs:
        context += "\n" + doc.page_content + "\n"

    # Update the global chat_history with the user's question
    global chat_history
    chat_history.append({"role": "user", "content": question})

    # Define prompt template
    prompt = f"""
    Here are the Task Context and History

    - Context: {context}
    - Chat History: {chat_history}
    - User Question: {question}
    """

    # Create the Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = ''
    # Generate the completion with the updated chat_history
    with client.messages.stream(
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="claude-3-haiku-20240307",
    ) as stream:
      for text in stream.text_stream:
          response += text
          yield response

    # Append the assistant's response to the chat_history
    chat_history.append({"role": "assistant", "content": response})

chat_interface = panel.chat.ChatInterface(
    callback=get_response, 
    callback_user="Sarathi",
    sizing_mode="stretch_width", 
    callback_exception='verbose',
    message_params=dict(
                default_avatars={"Sarathi": "S", "User": "U"}, #👽, 🥸
                reaction_icons={"like": "thumb-up"},
            ),
)

chat_interface.send(
    {"user": "Sarathi", "value": '''Welcome to Sarathi, your personal assistant for Assam Tourism.'''},
    respond=False,
)

template = panel.template.BootstrapTemplate(title="Sarathi", favicon="favicon.png", header_background = "#000000", main=[panel.Tabs( ('Chat', chat_interface), dynamic=True )])

template.servable()


