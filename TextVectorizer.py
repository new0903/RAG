
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union,List



class TextVectorizer:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
       # self.model_transform=SentenceTransformer(model_name)
        try:
            self.model_transform = SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Не удалось загрузить модель {model_name}: {e}")
    def encode_text(self,text:Union[str,List[str]])->np.ndarray:
        if isinstance(text,str):
            text=[text]
        embeddings = self.model_transform.encode(text, 
                                      convert_to_numpy=True,
                                      show_progress_bar=False,
                                      normalize_embeddings=True)
        return embeddings

    def get_vector_dimension(self,text:str)->int:
        """Получение размерности векторов модели"""
        if text is None:
            # Можно получить размерность без кодирования текста
            # Для этого закодируем тестовый текст
            text = "test"
        test_vector= self.encode_text(text)
        return test_vector[1]




