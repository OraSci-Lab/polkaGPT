from PyPDF2 import PdfReader
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import time 

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
folder_embedding="HuggingFace_embedding_Polkadot"
folder_data="/content/drive/MyDrive/VBI/RustGPT/books/Polkadot"     
def read_file(file_path, count):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # get passages (length of each passage is 250 words)
    batch_size=250
    list_texts=[]
    for i in range(0,len(text.split()),batch_size):
        end=i+batch_size+1
        #print("\n========================\nstart:           ", i)
        #print("end:             ", end)
        list_texts.append(" ".join(text.split()[i:end]))
    print(f"len file {count}:  ",len(list_texts))
    return list_texts
def get_list_text():
# path of data folder
    folder_path = folder_data  

    # get file in folder data
    file_list = os.listdir(folder_path)
    #print(file_list)
    list_doc=[]
    # get path of files
    for i in range(len(file_list)):
        print(f"file {i}")
        file_path = os.path.join(folder_path, file_list[i])
        file_text=read_file(file_path,i+1)
        for j in file_text:
            list_doc.append(j)
        print("len list_doc:   ", len(list_doc))

    return list_doc

def get_vector_db(texts):
    print("tới đyây rồi")
    db=FAISS.from_texts(texts,embedding)
    print("tới đây rồi")
    db.save_local(folder_embedding)
    return db

def main():
    list_text=get_list_text()
    print(type(list_text),"\n=============\n", list_text[-1],"==============")
    get_vector_db(list_text)
if __name__ == "__main__":
    print("start....")
    main()
    print("end...")
