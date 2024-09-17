import streamlit as st
from langchain_openai import AzureChatOpenAI
import os
# from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def initialize_openai():
    return AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment="gpt-4o",
        openai_api_key="8f74251696ce45698eb95495269f3d8c",
        azure_endpoint="https://cs-lab-azureopenai-gpt4.openai.azure.com/",
        temperature=0
    )

def process_document(bytes_data):
    # Load the document
    bytes_data = uploaded_file.read()
    f = open(uploaded_file.name, "wb")
    f.write(bytes_data)
    f.close()
    loader = PyPDFLoader(uploaded_file.name)
    data = loader.load()

    # Split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    return docs

def extract_entities(docs):
    # Initialize the LLM
    llm = initialize_openai()
    system_prompt = ("You are an expert Legal Assistant specializing in the Law domain. "
                        "You are skilled at entity extraction, identification, and relationship mapping, "
                        "with a strong focus on metadata accuracy. You are adept at helping people analyze "
                        "complex Legal documents to identify clauses as per the legal terms")

    UNTYPED_ENTITY_RELATIONSHIPS_JSON_GENERATION_PROMPT = """
    -Goal-
    Given a Legal document that is potentially relevant to this activity, identify all entities needed from the text in order to capture the information and ideas in the text..

    Steps:
    Entities needs to be extracted:
    1 Identify below given entities. For each identified entity, extract all the clauses related to identity:
    1st entity -  
    - entity_name: Blanket Indemnity Coverage
    - entity_description: Blanket Indemnity Coverage is where one party agrees to compensate the other party for any and all claims, losses, or damages arising from a contract or agreement. The description is highlighting that this indemnity is broad and comprehensive, covering a wide range of potential liabilities, unlike limited indemnity which is restricted to specific circumstances.
    2nd entity -  
    - entity_name: Uncapped Liability
    - entity_description: Uncapped Liability refers to a situation where there is no limit to the amount of liability that one party can be held responsible for under a contract. If an issue arises, the party with uncapped liability could potentially be responsible for paying an unlimited amount in damages or losses, depending on the severity of the breach or damage.
    3rd entity -  
    - entity_name: Termination Without Notice and Liability:
    - entity_description: Termination Without Notice and Liability refers to the right of one party to terminate a contract immediately, without providing advance notice to the other party and without being liable for any damages or penalties. This is often included in contracts to allow for quick exits in situations where continuing the relationship could be harmful or impractical. 
    4th entity -  
    - entity_name: Any Kind of Penalty:
    - entity_description: Any Kind of Penalty refer to clauses that impose additional financial obligations on a party for failing to fulfill their contractual obligations. This could include fines, interest payments, or other monetary sanctions designed to incentivize compliance with the contract.
    5th entity -  
    - entity_name: Post-Termination/Expiry Liability
    - entity_description: Post-Termination/Expiry Liability refers to the obligations or liabilities that remain in effect even after a contract has been terminated or has expired. Some contractual terms, such as confidentiality agreements, indemnities, or warranties, may continue to impose duties or liabilities on the parties after the official end of the contract. 

    2. From the entities identified in step 1, identify all the key clauses as much as possible from the document that are *clearly related* to Entity.
    For entities, extract the following information:
    - list each and every key clauses along with their number and text only if they strictly fall under the particular entity from the document, if you don't find any clause under that particluar entity, don't return true negatives, strictly don't return any clause, only return "No clause found under this entity"
    - Reason for the key clause to fall under that particular entity

    3. Strictly Return output having the clauses only if they fall under the particular entities identified. If you have to translate, just translate the descriptions, nothing else!
    
    - stick to the task assigned and don't answer for the task not asked

    - Identify only those clauses which falls in the entity strictly as per the description. Don't return any possiblity as the clauses should be defined with respect to legal terminology.

    - If you find no clause is found with respect to Any particular entity, just return "no clause found" under that enitity. Don't try to list any clause for that entity. 

    ######################
    -Real Data-
    ######################
    Text: {input_text}
    ######################
    Output:
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": UNTYPED_ENTITY_RELATIONSHIPS_JSON_GENERATION_PROMPT.format(input_text=docs)}
    ]






    response = llm.invoke(messages)
    return response.content.split('\n')



# Streamlit app UI
st.title("Legal Document Entity Extractor")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    st.write("Processing the document...")
    # Process the uploaded document
    docs = process_document(uploaded_file)
    
    # Extract entities
    output = extract_entities(docs)
    
    # Display the output
    st.write("### Extracted Clauses and Entities:")
    for line in output:
        st.write(line)

