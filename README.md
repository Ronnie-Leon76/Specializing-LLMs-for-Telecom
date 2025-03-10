# Specializing LLMs for Telecom Networks

- While LLMs have undeniably demonstrated their prowess across diverse sectors, their integration into the telecommunications industry has been somewhat limited. However, this landscape is undergoing a gradual metamorphosis as researchers delve deeper into the potential of LLMs within this domain. With this competition, our objective is to tackle this challenge and pave the way for the development of telecom GPTs.

## Requirements
- Python 3.10
- Dependencies listed in `requirements.txt`

## Technologies Used
- LangChain


## Setup
### To run this solution, follow the following instructions:

1. Create a new virtual environment:
    - If using vanilla [virtualenv](https://virtualenv.pypa.io/en/latest/), run `virtualenv venv` and then activate the virtual environment with the following commands for linux `source venv/bin/activate` and for a Windows machine `<venv>\Scripts\activate.bat`
    - If using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.org/en/latest/), run `mkvirtualenv venv`
    - Before installing the project required dependencies, ensure you run the following commands to install some global dependencies required by the project:
      ```bash
      sudo apt install tesseract-ocr -y
      ```
      ```bash
      sudo apt install libtesseract-dev -y
      ```
      ```bash
      sudo apt-get install poppler-utils -y
      ```

2. Install the requirements with `pip install -r requirements.txt`

3. Start Qdrant in a docker container using the following command:
    ```bash
    docker run -d -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
    ```
4. Once you have everything setup in your environment, run `python main.py ./Data/ TeleQnA_testing1_refined.csv` to run the solution.

### To run the LlamaIndex solution version, follow the following instructions:

1. Create a new virtual environment:
    - If using vanilla [virtualenv](https://virtualenv.pypa.io/en/latest/), run `virtualenv venv` and then activate the virtual environment with the following commands for linux `source venv/bin/activate` and for a Windows machine `<venv>\Scripts\activate.bat`
    - If using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.org/en/latest/), run `mkvirtualenv venv`
  
2. Install the requirements with `pip install -r llama_index_requirements.txt`

3. First create a LlamaIndex RAG pipeline with training data for finetuning using the following command: ```python vector_store_for_rag.py```

4. Then fine-tune Microsoft Phi-2 using the training dataset created using the following command: ```python fine_tuning.py```

5. Run the following command: ```python llama_index_main.py``` to run inference to generate a submission file.
