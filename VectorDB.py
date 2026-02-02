
import sqlite3
import numpy as np
import json
from typing import List,Dict,Any

from sentence_transformers.util import similarity


class VectorDB:
    def create_table(self):
        cursor=self.connection.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_text ON documents(text)
        ''')
        self.connection.commit()
        cursor.close()
    #    self.connection.close()

    def __init__(self,path:str="vectors.db"):
        self.connection=sqlite3.connect(path)
        self.create_table();
    

    def insert_vector(self,text:str,vector:np.ndarray):#, metadata: Dict = None
        """
        Вставка вектора в базу данных.
        """
        vector_bytes = vector.astype(np.float32).tobytes()
       
        # metadata_json=json.dumps(metadata) if metadata else None

        cursor=self.connection.cursor();

        cursor.execute('''
            Insert into documents (text, embedding) 
            VALUES (?, ?)
        ''',(text,vector_bytes))

        self.connection.commit()
        last_row_id=cursor.lastrowid
        cursor.close()
       # self.connection.close()
        return last_row_id

    def search_simular(self,query_vector:np.ndarray,limit:int=5):
        cursor=self.connection.cursor()

        cursor.execute('''
            SELECT id, text, embedding FROM documents
        ''')
        result=[]
        query_norm=np.linalg.norm(query_vector)

        for row in cursor.fetchall():
            doc_id, text, vector_bytes = row

            stored_vector = np.frombuffer(vector_bytes, dtype=np.float32)
            stored_norm=np.linalg.norm(stored_vector)

            cosine_sim=np.dot(query_vector,stored_vector)

            if query_norm==0 or stored_norm==0:
                similarity=0
            else:
                similarity=cosine_sim/(query_norm*stored_norm)



            result.append({
                'id': doc_id,
                'text': text,
                'similarity': similarity[0]
            })

        cursor.close()
    #   self.connection.close()
        result.sort(key=lambda x: x['similarity'], reverse=True)
        return result[:limit]





