import sys
import os
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class chatbot():

  def __init__(self, link, embeddings, link_store):

    self.documents = self.getDocument(link)
    self.embeddings = embeddings
    self.store = self.createStore(self.documents, self.embeddings, link_store)

  def getDocument(self, link):
    with open(link, "rb") as f:
      doc = pickle.load(f)
    return doc

  def createStore(self,doc, embeddings, link):
    try:
          store = Qdrant.from_documents(
              documents=doc,
              embedding=embeddings,
              path=link,
              collection_name="vector_db",
              force_recreate=True
          )
          return store
    except Exception as e:
        print(f"Lỗi khi tạo Qdrant store: {str(e)}")
        return None

  

  def retriever_similar(self, query, k):
    store = self.store
    retriever = store.as_retriever(
          search_type="similarity_score_threshold",
          search_kwargs={
              "k": k,
              "score_threshold": 0.6,
              "filter": None
          }
      )
    return retriever.invoke(query)

  def format_context(self, docs):
    text = ''
    for doc in docs:
        text += doc.page_content + '\n'
    return text
