
import chromadb 
from chromadb.utils import embedding_functions
from typing import Union,List
import datetime
from PDFReader import get_pdf_files_directory, extract_text_by_page_pdfplumber


CHROMA_DATA_PATH = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"
# timenow=datetime.date.isoformat("YYYY-MM-DD")
now = datetime.datetime.now()#.today()  # объект datetime
timenow = now.isoformat()  # "2025-01-14T15:30:45.123456"

class VectorChromaDB:
  



# {
# "title":"описание/заголовок документа",
# "published":datetime.date().isoformat(),
# "page_num":123
# "article":"описание конкретного абзаца"
# "source":"ссылка на документ или путь к нему или наименование дока"
# } 

# chunks.append({
#     "text": p_text,
#     "page": page_num,
#     "date_create":datetime.datetime.isoformat(),
#     "name_dock":pdf_path
# })       


    def insertData(self,text:Union[str,List[str]],metadata:List):       
       if isinstance(text,str):
            text=[text]

       self.collction.add(
            documents=text,
            ids=[f"id{i}" for i in range(len(text))],
            metadatas=metadata,
            )

    def findDocument(self,query_text:str):
        query_results=self.collction.query(
            query_texts=[query_text],
            n_results=5
            )
        result_array=[]
        
#     # Получаем количество результатов
#     num_results = len(result["documents"][0])
        num_results=len(query_results["documents"][0])
        for i in range(num_results):
            result_dict = {
                "id": query_results["ids"][0][i],
                "document": query_results["documents"][0][i],
                "distances": query_results["distances"][0][i],
                "metadata":query_results["metadatas"][0][i]
            }
            result_array.append(result_dict)
        return result_array

    def seed_date(self):
        files_pdf= get_pdf_files_directory("./docs/")
        result=[]

        for path in files_pdf:
           paragraphs= extract_text_by_page_pdfplumber(path)
           for para in paragraphs:
                result.append(para)


        texts=[item["text"] for item in result]
        meta=[{"page":item["page"],"date_create":item["date_create"],"name_dock":item["name_dock"]} for item in result]

        self.insertData(texts,meta)


    def __init__(self,name_collection:str=COLLECTION_NAME,pathdb:str=CHROMA_DATA_PATH,model_name_db:str=EMBED_MODEL):
        
        print("PersistentClient init")
        self.client=chromadb.PersistentClient(path=pathdb)  #chromadb.  #.Client(database="./chroma_db")
        print("SentenceTransformerEmbeddingFunction init")
        self.embedding_fnc=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name_db)
        print("get_or_create_collection init")
        self.collction=self.client.get_or_create_collection(name=name_collection,embedding_function=self.embedding_fnc)
        if self.collction.count() == 0:
            print("seed_date init")
            self.seed_date()
        


# query_results = collection.query(
#     query_texts=["Find me some delicious food!"],
#     n_results=1,
# )

# query_results.keys()
# dict_keys(['ids', 'distances', 'metadatas', 'embeddings', 'documents'])

# query_results["documents"]
# [['Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.']]

# query_results["ids"]
# [['id3']]

# query_results["distances"]
# [[0.7638263782124082]]

# query_results["metadatas"]
# [[{'genre': 'food'}]]