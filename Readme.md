

### Инструкция по запуску

### Пакеты для установки:

text
pip install PyMuPDF chromadb streamlit
Дополнительно можно установить sentence-transformers:

text
pip install sentence-transformers
Итог:

text
pip install PyMuPDF chromadb streamlit sentence-transformers
### Настройка поиска по тексту:
Перед первым запуском сложите все PDF-документы в папку docs.

Если позже вы добавите новые PDF-файлы, удалите всё из папки chroma_db.

Если не хотите использовать Chroma, используйте TextVectorizer, VectorDB и PDFReader.

TextVectorizer — преобразует текст в векторы.

VectorDB — создает SQLite базу данных, сохраняет туда текст с векторами и осуществляет поиск.

Короче, здесь надо будет написать пару строк кода, чтобы соединить всё это.

### Настройка OpenAI:
В sseapi вставить свои model и url.

В функции request, в запросе response = await session.post(...) настройте:

python
headers={
    "Content-Type": "application/json",
    # "Authorization": "" // добавить свой токен
}
Короче говоря, настройте заголовки и тело запроса так, как вам надо.

Для того чтобы всё работало как надо, советую установить и зарегистрироваться в Ollama. Их API точно будет работать, у них можно использовать локальные и облачные модели и т.д.

После того как вы всё настроите, вот команда для запуска:
### streamlit run PythonAppSearchText.py



