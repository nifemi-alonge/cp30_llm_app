# packages # 

import streamlit as st
import pandas as pd 
import numpy as np
from numpy.linalg import norm
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

st.title("Clean Power 2030 LLM App")

# functions #
def get_embedding(text):
   text = text.replace("n", " ")
   return embeddings.embed_documents([text])[0]


# Inputs #
context_source = st.text_input("Type your context source URL here", 
                       value = "https://www.gov.uk/government/publications/clean-power-2030-action-plan/clean-power-2030-action-plan-a-new-era-of-clean-electricity-main-report")

passkey = st.text_input("Your API token", 'xxxxxxxxxxxx',type="password")

model_name = st.text_input("Type your Hugging Face embeddings model of choice", value = "sentence-transformers/all-mpnet-base-v2")

users_question = st.text_input("Type your question here")

# Process context source data #
loader = WebBaseLoader(context_source)
docs = loader.load()
article_text = docs[0].page_content

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 105,
    chunk_overlap  = 20,
    length_function = len,
)

texts = text_splitter.create_documents([article_text])

st.subheader("Embeddings generator")

# create embeddings 
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# process dataframe 
text_chunks = []

for text in texts:
    text_chunks.append(text.page_content)

df = pd.DataFrame({'text_chunks': text_chunks})

if st.button("convert context source to embeddings and compare"):
    df['hf_embedding'] = df.text_chunks.apply(lambda x: get_embedding(x))
    st.write(df.head())
    # compare user 
    question_embedding = get_embedding(text=users_question)

    # create a list to store the calculated cosine similarity
    cos_sim = []

    for index, row in df.iterrows():
        A = row.hf_embedding
        B = question_embedding

        # calculate the cosine similarity
        cosine = np.dot(A,B)/(norm(A)*norm(B))

        cos_sim.append(cosine)

    df["cos_sim"] = cos_sim
    df = df.sort_values(by=["cos_sim"], ascending=False)
    st.write(df.head())

st.subheader("Send Question to LLM")

repo_id = st.text_input("repo id of llm model in HF",value="mistralai/Mistral-Nemo-Base-2407")

temperature = st.number_input("temp of llm repsonse", min_value=0, max_value=1, step=0.1, value=0.7)

col1, col2 = st.columns(2)
# no context
with col1:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        huggingfacehub_api_token=passkey,
        provider="auto",
    )
    if st.button("Question LLM no context"):
        st.write(llm(users_question)) # compare before and after

# with context
with col2:
    # define the context for the prompt by joining the most relevant text chunks
    context = ""

    for index, row in df[0:50].iterrows():
        context = context + " " + row.text_chunks

    # define the prompt template
    template = """
    You are a chat bot who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly writting in the documentation, say "Sorry, I don't know how to help with that."
    Give a short response.

    Context sections:
    {context}

    Question:
    {users_question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])

    # fill the prompt template
    prompt_text = prompt.format(context = context, users_question = users_question)

    if st.button("Send question with context"):
        st.write(llm(prompt_text))