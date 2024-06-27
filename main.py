import argparse
import os, sys
import warnings
import pandas as pd
import pickle
import re
from tqdm.auto import tqdm
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from Ingestion.ingest import extract_text_and_metadata_from_docx_document


warnings.filterwarnings("ignore")

_ = load_dotenv(find_dotenv())
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

COLLECTION_NAME = 'Telecom-Collection'
QDRANT_URL = 'http://localhost:6333'


TEST_DATASET_ALPACA_PROMPT = """Below is an instruction that describes a task.
Use the following pieces of context to answer the question by selecting the option that has the right answer..
{}

### Question:
{}

### Option 1:
{}

### Option 2:
{}

### Option 3:
{}

### Option 4:
{}

### Option 5:
{}

### Category:
{}

Provide the answer in the following format:

### Answer:
option [number]: [option answer]
"""

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    #attn_implementation="flash_attention_2", # if you have an ampere GPU
    token=HF_TOKEN
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.1, device_map='auto')
llm = HuggingFacePipeline(pipeline=pipe)


# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={'device': 'cuda:0'})


def create_vector_db(documents, embedding_model):
    qdrant = Qdrant.from_documents(
        documents,
        embedding_model,
        url=QDRANT_URL,
        prefer_grpc=False,
        # api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )
    print(f"Created vector database with {len(documents)} documents")

def initialize_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def initialize_bm25_retriever(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    # Save bm25_retriever as a pickle file
    with open('bm25_retriever.pkl', 'wb') as f:
        pickle.dump(bm25_retriever, f)
    return bm25_retriever

def load_bm25_retriever():
    with open('bm25_retriever.pkl', 'rb') as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever

def retrieval_qa_chain(llm,retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain

def regex_query_response_postprocessor(response):
    pattern = r'Answer: option (\d+):'
    match = re.search(pattern, response)
    if match:
        answer_option = match.group(1)
    else:
        answer_option = None
    return answer_option

def extract_question_id_from_test_df_index(index):
    pattern = r'question (\d+)'
    match = re.search(pattern, index)
    if match:
        question_id = match.group(1)
    else:
        question_id = None
    return question_id

def regex_query_response_postprocessor(response):
    pattern = r'Answer:\s*option\s*(\d+):'
    match = re.search(pattern, response)
    if match:
        answer_option = match.group(1)
    else:
        answer_option = 0
    return answer_option

def extract_question_id_from_test_df_index(index):
    pattern = r'question (\d+)'
    match = re.search(pattern, index)
    if match:
        question_id = match.group(1)
    else:
        question_id = None
    return question_id



def main():
    parser = argparse.ArgumentParser(description='Partition DOCX files in a directory')
    parser.add_argument('dir_path', type=str, help='Path to the directory containing DOCX files')
    parser.add_argument('csv_path', type=str, help='Path to the TeleQnA_testing1.csv file')

    args = parser.parse_args()

    dir_path = args.dir_path
    csv_path = args.csv_path

    if not os.path.exists(dir_path):
        print(f"DOCX Documents Directory path {dir_path} does not exist")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"TeleQnA_testing1 CSV path {csv_path} does not exist")
        sys.exit(1)

    bm25_retriever = None

    try:
        client = initialize_qdrant_client()
        embedding_model = load_embedding_model()
        if not client.collection_exists(COLLECTION_NAME):
            docx_files = [f for f in os.listdir(dir_path) if f.endswith('.docx')]

            documents = []
            for docx_file in tqdm(docx_files, desc='Processing DOCX files'):
                docx_path = os.path.join(dir_path, docx_file)
                try:
                    df = extract_text_and_metadata_from_docx_document(docx_path)
                    print(f"Extracted text and metadata from {docx_file}")
                    for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
                        parent_id = row['Parent_Id']
                        file_name = row['Filename']
                        text = row['Text']
                        page_number = row['Page_Number']
                        document = Document(
                            page_content = text,
                            metadata = {
                                'id': str(index) + '_' + str(parent_id) + '_' + file_name + '_' + str(page_number),
                                'type': 'text',
                                'filename': file_name,
                                'page_number': page_number
                            }
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"Error processing {docx_file}: {str(e)}")
            create_vector_db(documents, embedding_model)
            bm25_retriever = initialize_bm25_retriever(documents)

        # Load the Qdrant vector store
        db = Qdrant(client=client, embeddings=embedding_model, collection_name=COLLECTION_NAME)
        qdrant_retriever = db.as_retriever(search_kwargs={'k': 5})
        # Load the BM25 retriever
        if not bm25_retriever:
            bm25_retriever = load_bm25_retriever()
        # Create an ensemble retriever with the BM25 and Qdrant retrievers
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, qdrant_retriever], weights=[0.5, 0.5])

        test_df = pd.read_csv(csv_path)
        sample_submission_df = pd.DataFrame(columns=['Question_ID', 'Answer_ID'])
        # qa_chain = retrieval_qa_chain(llm, ensemble_retriever)
        # llm_chain = LLMChain(llm=llm)

        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing rows'):
            question = row['question']
            option1 = row['option 1']
            option2 = row['option 2']
            option3 = row['option 3']
            option4 = row['option 4']
            option5 = row['option 5']
            category = row['category']
            test_index = row['Test_Index']
            results = ensemble_retriever.get_relevant_documents(question)
            query = TEST_DATASET_ALPACA_PROMPT.format(results, question, option1, option2, option3, option4, option5, category)
            response = llm.invoke(query)
            # print(response)
            # value = response[0]['generated_text']
            # print(value)
            answer_id = regex_query_response_postprocessor(response)
            question_id = extract_question_id_from_test_df_index(test_index)
            sample_submission_df = pd.concat([
                sample_submission_df,
                pd.DataFrame([{
                    'Question_ID': question_id,
                    'Answer_ID': answer_id
                }])
            ], ignore_index=True)

        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        sample_submission_df.to_csv(f'sample_submission_{timestamp}.csv', index=False)
        print(f"Sample submission file saved as sample_submission_{timestamp}.csv")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
        
