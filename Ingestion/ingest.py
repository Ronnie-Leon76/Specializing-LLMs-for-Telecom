import pandas as pd
from unstructured.partition.docx import partition_docx
from unstructured.cleaners.core import clean

def extract_text_and_metadata_from_docx_document(docx_path):
    """
    Extracts text and metadata from a docx document
    :param docx_path: path to the docx document
    :return: pandas dataframe with extracted text and metadata
    """
    elements = partition_docx(docx_path)

    data = []
    for c in elements:
        row = {}
        row['Element_Type'] = type(c).__name__
        row['Filename'] = c.metadata.filename
        row['Date_Modified'] = c.metadata.last_modified
        row['Filetype'] = c.metadata.filetype
        row['Parent_Id'] = c.metadata.parent_id
        row['Category_Depth'] = c.metadata.category_depth
        row['Page_Number'] = c.metadata.page_number
        row['Text'] = clean(c.text, bullets=True, extra_whitespace=True, dashes=True, trailing_punctuation=True)
        data.append(row)

    df = pd.DataFrame(data)
    return df
