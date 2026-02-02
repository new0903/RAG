


### Инструкция по запуску


### Пакеты к установке:


pip install PyMuPDF
pip install chromadb
pip import streamlit

дополнительно можно sentence-transformers
pip import sentence-transformers


### итог 
pip install PyMuPDF chromadb streamlit sentence-transfor


### Настройка поиска по тексту:

Перед первым запуском все пдф документы складываем в папку docs

После если вы добавите ещё файлы пдф тогда удалите всё из папки chroma_db

Если не хотите использовать chroma тогда Используйте TextVectorizer, VectorDB и PDFReader

TextVectorizer - преобразует текст в векторы 

VectorDB - создает sqlite бд и сохраняет туда текст с векторами, а так же осуществляет поиск по векторам.

короче здесь надо будет написать пару строк кода что бы соеденить всё это.


### Настройка OpenAI

в sseapi вставить свои model и url

в функции request в запрос response=await session.post(
                    url,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json"#,"Authorize":""//добавить свой токен 
                    }
                ) 

короче говоря настройте заголовки и тело запрос так как вам надо.

для того что бы все работало как надо советую установить и зарегестрироваться в Ollama. Их api точно будет работать,
у них можно использовать локальные и облачные модели и т.д. 



После того как вы все настроите  

вот команда для запуска

### streamlit run PythonAppSearchText.py



