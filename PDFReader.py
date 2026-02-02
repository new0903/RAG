  
#import pdfplumber
import fitz
import os
import datetime


def extract_text_by_page_pdfplumber(pdf_path):
    """Извлечение текста с лучшим сохранением форматирования"""
    chunks = []
    
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf, 1):
            #text = page.extract_text_simple()
            # text = page.extract_text_lines()
            
            text = page.get_text()
            # text = page.extract_text(
            #     x_tolerance=1,    # Чувствительность к горизонтальным пробелам
            #     y_tolerance=1,    # Чувствительность к вертикальным пробелам
            #     layout=True,     # False обычно лучше для обычного текста
            #     extra_attrs=[]    # Не включать атрибуты
            #     )
        


            if text and text.strip():
                # Разбиваем текст на абзацы.replace('\n'," \n")
                text=text.split('\n')
                lines_text = [p.strip() for p in text if p.strip()]
                paragraphs=[]
                current_p=[]
                for i in range(0,len(lines_text)):
                    line_t=lines_text[i]
                    is_end_of_sentence = line_t.strip().endswith(('.', '!', '?', ':"', '."'))
                    current_p.append(line_t)
                 

                    if (is_end_of_sentence and len(current_p)>2) or len(current_p)>8:
                        p_text=" ".join(current_p)
                        current_p=[]
                        now = datetime.datetime.now()  # объект datetime
                        timenow = now.isoformat()  # "2025-01-14T15:30:45.123456"
                        chunks.append({
                            "text": p_text,
                            "page": page_num,
                            "date_create":timenow,
                            "name_dock":pdf_path
                        })       

    return chunks

def get_pdf_files_directory(path:str):
    pdfs=[]
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(path, file))
    return pdfs






# Использование
# pdfsd = get_pdf_files_directory('/path/to/directory')
# print(f"Найдено {len(pdfsd)} PDF файлов:")
# print(pdfsd)


# chunks = extract_text_by_page_pdfplumber("test.pdf")

# print(chunks)
