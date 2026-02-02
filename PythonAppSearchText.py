# Дополнительные параметры сохранения → Кодировка
# UTF-8 with signature 65001
print("старт")

import asyncio 

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# from sentence_transformers import SentenceTransformer
# from typing import List
import json
import time
import aiohttp

#from TextVectorizer import TextVectorizer
#from VectorDB import VectorDB
from VectorChromaDB import VectorChromaDB
from sseapi import SSE_Bot


bot=SSE_Bot()

print("bot inited")

documents=VectorChromaDB()

print("модель загружена")


st.title("💬 Чат-поисковик документов")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "lock" not in st.session_state:
    st.session_state.lock = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Задайте вопрос о документах..."):
    if not st.session_state.lock:
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.chat_input("Задайте вопрос о документах...",disabled= st.session_state.lock)
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            def run_async_process():
                async def process_request():  
                    full_response=""
                   
                    message_assist={
                        "role": "assistant", 
                        "content": "🤔 Нет данных для ответа"
                    }
                    message_assist["content"]="🔍 Анализирую запрос..."
                    st.session_state.messages.append(message_assist)
                    #nonlocal full_response
                    message_placeholder.markdown(message_assist["content"])
                    find_documents=documents.findDocument(prompt)
                    print(find_documents)
                    try:
                        json_dock=json.dumps(find_documents,ensure_ascii=False)
                        response,session=await bot.request(prompt,json_dock)
                    except Exception as e:
                        message_placeholder.markdown(f"❌ Ошибка при запросе: {e}")
                        return f"❌ Ошибка при запросе: {e}"
            
 
                    try:       
                        await asyncio.sleep(1)
                      #  print("начинаю читать сообщения")
            
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                         #   print("цикл for",line)
                            text_line=bot.procces_message_line(line)
                         #   print("text_line",text_line)
                            if text_line=="[none]":
                                await asyncio.sleep(1)
                                continue
                            if text_line in ["[null]","[done]"]:
                                break
                            full_response+=text_line
                            message_assist["content"]=full_response
                            message_placeholder.markdown(message_assist["content"])
                            await asyncio.sleep(1)
                        # # st.session_state.messages.append({
                        # #     "role": "assistant", 
                        # #     "content": full_response
                        # # })
                    except aiohttp.ClientConnectionError as e:
                        st.warning(f"Соединение прервано: {e}")
                    except RuntimeError as e:
                        print(e)
                    finally:
                        if response and not response.closed:
                            response.close()
                        if session and not session.closed:
                            await session.close()
                    if full_response:
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response
                        })
                    else:
                        message_placeholder.markdown("🤔 Нет данных для ответа")
                        
                    # st.chat_input("Задайте вопрос о документах...",disabled= st.session_state.lock)
                    
                    return full_response if full_response else "🤔 Нет данных для ответа"

                return asyncio.run(process_request())
            

            try:
                res=run_async_process()
            
                #request
            except RuntimeError as e:
                print(e)











                
# example_texts = [
#     "Искусственный интеллект меняет мир вокруг нас",
#     "Глубокое обучение является подразделом машинного обучения",
#     "Нейронные сети имитируют работу человеческого мозга",
#     "Python - популярный язык для анализа данных",
#     "Векторное представление текста - ключевая задача NLP",
#     "Привет, как дела?",
#     "Сегодня отличная погода",
#     "Машинное обучение очень интересно",
#     "Hello, how are you?",
#     "Сегодня я изучаю Python программирование",
#     "Явление, которое Эйнштейн называл «spooky action at a distance», где состояния двух частиц остаются взаимосвязанными независимо от расстояния, становится краеугольным камнем квантовых технологий. Квантовая запутанность лежит в основе принципов будущих квантовых компьютеров, способных решать задачи, недоступные классическим машинам, а также сверхзащищённых линий связи (квантовая криптография). Хотя практическая реализация сталкивается с трудностями поддержания когерентности, прогресс в этой области может изменить вычислительную технику и кибербезопасность в ближайшие десятилетия.",
#     """
#     Феномен медленного туризма
# В противовес «галочкам» у достопримечательностей набирает популярность философия slow travel, делающая акцент на глубоком погружении в локальную культуру и окружающую среду. Путешественники арендуют жильё на месяц, изучают язык, участвуют в жизни сообщества и минимизируют авиаперелёты в пользу поездов и велосипедов. Этот подход не только снижает экологический след, но и трансформирует сам опыт поездки из потребительского акта в процесс образования и личностного роста.
#     """,
#     """
#     Бионика и протезирование нового поколения
# Современные протезы перестали быть пассивными макетами, а превратились в бионические устройства, управляемые сигналами мозга или остаточной мускулатуры. Сенсоры позволяют ощущать текстуру поверхности и температуру, а machine learning algorithms помогают плавно адаптировать движения под задачи пользователя. Эти технологии не только возвращают людям базовые функции, но и открывают дискуссию о human enhancement — добровольном улучшении физических возможностей организма.
#     """,
#     """
#     Психология принятия решений
#     Исследования Даниэля Канемана и Амоса Тверски показали, что человеческое мышление подвержено систематическим когнитивным искажениям — эвристикам. Например, «эффект якоря» или «склонность к подтверждению своей точки зрения» влияют на наши финансовые выборы, политические взгляды и бытовые суждения. Понимание этих механизмов помогает создавать более эффективные публичные политики и интерфейсы, которые nudging-методами мягко подталкивают людей к оптимальным решениям.
#     """,
#     """
#  Ренессанс настольных игр
# В эпоху доминирования цифровых развлечений настольные игры переживают неожиданный бум. От сложных eurogames с экономическими стратегиями до кооперативных приключений — этот формат предлагает тактильное взаимодействие и живое социальное общение, которого часто не хватает в онлайн-среде. Кроссоверы с популярными франшизами, краудфандинг платформ вроде Kickstarter и развитие коммьюнити превратили хобби в многомиллиардную индустрию. 
#     """,
#     """
#    Урбанизация и мегаполисы будущего
# К 2050 году более двух третей населения Земли будет проживать в городах, что создаст беспрецедентную нагрузку на инфраструктуру, экологию и социальные службы. Концепции «умных городов» предлагают использовать big data, IoT и автоматизацию для оптимизации traffic flow, энергопотребления и утилизации отходов. Однако ключевым вызовом остаётся обеспечение не только технологической эффективности, но и социальной инклюзивности, уменьшения неравенства и сохранения зелёных пространств. 
#     """,
#     """
#     Искусственный интеллект в творчестве
# Нейросети, такие как DALL-E и Midjourney, способны генерировать изображения по текстовым описаниям, стирая грань между человеческим и машинным творчеством. Это вызывает живые споры о природе искусства, авторском праве и будущем профессий в дизайне и визуальной сфере. Хотя ИИ пока не обладает сознанием и интенцией, его способность комбинировать стили и создавать эстетически pleasing работы заставляет пересматривать само определение креативности.
#    """,
# ]
# test.insertData(example_texts,[])


# print("данные загружены")

#resulttest=test.findDocument("обучение и психология развития решений ИИ ");

# context=VectorDB()
# #streamlit run PythonAppSearchText.py
# vectorizer = TextVectorizer()


# for record in example_texts:
#     e=vectorizer.encode_text(record)
#     context.insert_vector(record,e)

#vectorizer.get_vector_dimension

# query_test=vectorizer.encode_text(" обучение очень и искусственный интеллект ")
# result_test= context.search_simular(query_test)




                   






# st.markdown('<h1 class="main-header">🔤 Преобразование текста в векторы</h1>', unsafe_allow_html=True)

# # Основной контейнер
# with st.container():
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Ввод текста
#         st.markdown('<h3 class="sub-header">📝 Введите текст для векторизации</h3>', unsafe_allow_html=True)
        
#         # Большое поле для ввода с примером
#         input_text = st.text_area(
#             "Введите текст (поддерживается кириллица):",
#             height=200,
#             placeholder="Например: Машинное обучение - это увлекательная область искусственного интеллекта, которая позволяет компьютерам учиться на данных без явного программирования.",
#             help="Можно вводить текст на русском, английском и других языках"
#         )
        
#         # Кнопки действий
#         col1_1, col1_2, col1_3 = st.columns(3)
#         with col1_1:
#             process_btn = st.button("🔮 Преобразовать в вектор", type="primary", use_container_width=True)
#         with col1_2:
#             clear_btn = st.button("🧹 Очистить", use_container_width=True)
#         with col1_3:
#             example_btn = st.button("📚 Поиск", use_container_width=True)
        
        
#         if clear_btn:
#             input_text = ""
#             st.rerun()
    
#     with col2:
#         # Информационная панель
#         st.markdown('<h3 class="sub-header">ℹ️ Информация</h3>', unsafe_allow_html=True)
        
#         with st.container():
#             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#          #   st.metric(label="Размерность вектора", value=vectorizer.get_vector_dimension())
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         st.markdown("""
#         <div style="background-color: #f0f9ff; padding: 15px; border-radius: 10px; margin-top: 20px;">
#             <h4>📊 О векторизации:</h4>
#             <p>Текст преобразуется в числовой вектор размерностью <strong>384</strong>. Каждое число представляет собой признак текста.</p>
#             <p>Модель: <code>paraphrase-multilingual-MiniLM-L12-v2</code></p>
#             <p>Поддерживает: русский, английский и другие языки</p>
#         </div>
#         """, unsafe_allow_html=True)



# if example_btn and input_text:
    
#     # query=vectorizer.encode_text(input_text)
#     # result= context.search_simular(query)
#     result=test.findDocument("обучение и психология развития решений ИИ ");
#     result_array = []

#     # Получаем количество результатов
#     num_results = len(result["documents"][0])

#     for i in range(num_results):
#         result_dict = {
#             "id": result["ids"][0][i],
#             "document": result["documents"][0][i],
#             "distances": result["distances"][0][i]
#         }
#         result_array.append(result_dict)

#     df = pd.DataFrame(result_array)
        
#     # Добавляем номер строки
#     df['№'] = range(1, len(df) + 1)
        
#     # Переупорядочиваем колонки
#     df = df[['№', 'id', 'document', 'distances']]
        
#     # Убедимся, что similarity - это числа (float)
#     df['distances'] = pd.to_numeric(df['distances'], errors='coerce')
        
#     # Форматируем сходство в проценты (только для отображения)
#     df['similarity_display'] = df['distances'].apply(lambda x: f"{x:.2%}")
        
#     # Форматируем сходство в проценты
#     df['distances'] = df['distances']
#     st.markdown('<h2 class="sub-header">📊 Результаты векторизации</h2>', unsafe_allow_html=True)
#     st.dataframe(
#         df,
#         column_config={
#             '№': st.column_config.NumberColumn(use_container_width=true),
#             'id': st.column_config.NumberColumn("ID документа"),
#             'document': st.column_config.TextColumn("Текст", use_container_width=true),
#             'distances': st.column_config.TextColumn("Сходство", use_container_width=true)
#         },
#         hide_index=True,
#         use_container_width=True
#     )    
#     # input_text = np.random.choice(example_texts)

#     #st.rerun()
# # Обработка введенного текста
# if process_btn and input_text:
#     with st.spinner("Выполняю векторизацию..."):
#         # Добавляем небольшую задержку для визуального эффекта
#         #time.sleep(0.5)
        
#         # Векторизация текста
#         #vector = vectorizer.encode_text(input_text)
        
#         # Отображение результатов
#         # st.markdown("---")
#         # st.markdown('<h2 class="sub-header">📊 Результаты векторизации</h2>', unsafe_allow_html=True)
        
#         # # Основные метрики
#         # col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        
#         # with col_metrics1:
#         #     st.metric("Длина текста", f"{len(input_text)} симв.")
        
#         # with col_metrics2:
#         #     st.metric("Количество слов", f"{len(input_text.split())}")
        
#         # with col_metrics3:
#         #     min_val = np.min(vector)
#         #     st.metric("Мин. значение", f"{min_val:.4f}")
        
#         # with col_metrics4:
#         #     max_val = np.max(vector)
#         #     st.metric("Макс. значение", f"{max_val:.4f}")
        
#         # # Визуализация вектора
#         # tab1, tab2, tab3 = st.tabs(["📈 График вектора", "🔢 Числовые значения", "📋 Детали"])
        
#         # with tab1:
#         #     # График первых 50 значений вектора
#         #     fig = go.Figure()
            
#         #     # Основной график
#         #     fig.add_trace(go.Scatter(
#         #         x=list(range(1, min(51, len(vector[0])) + 1)),
#         #         y=vector[0][:50],
#         #         mode='lines+markers',
#         #         name='Значения вектора',
#         #         line=dict(color='#3b82f6', width=2),
#         #         marker=dict(size=8, color='#1d4ed8')
#         #     ))
            
#         #     # Средняя линия
#         #     fig.add_hline(y=np.mean(vector[0]), 
#         #                  line_dash="dash", 
#         #                  line_color="red",
#         #                  annotation_text=f"Среднее: {np.mean(vector[0]):.4f}")
            
#         #     fig.update_layout(
#         #         title="Первые 50 значений вектора",
#         #         xaxis_title="Индекс значения",
#         #         yaxis_title="Значение",
#         #         height=400,
#         #         showlegend=True
#         #     )
            
#         #     st.plotly_chart(fig, use_container_width=True)
            
#         #     # Гистограмма распределения
#         #     fig2 = px.histogram(
#         #         x=vector[0],
#         #         nbins=30,
#         #         title="Распределение значений вектора",
#         #         labels={'x': 'Значение', 'y': 'Количество'},
#         #         color_discrete_sequence=['#3b82f6']
#         #     )
#         #     fig2.update_layout(height=300)
#         #     st.plotly_chart(fig2, use_container_width=True)
        
#         # with tab2:
#         #     # Числовое представление
#         #     st.markdown("**Первые 20 значений вектора:**")
            
#         #     # Создаем DataFrame для красивого отображения
#         #     data = []
#         #     for i, val in enumerate(vector[0][:20], 1):
#         #         data.append({
#         #             "Индекс": i,
#         #             "Значение": val,
#         #             "Абсолютное значение": abs(val)
#         #         })
            
#         #     df = pd.DataFrame(data)
#         #     st.dataframe(df, use_container_width=True, hide_index=True)
            
#         #     # Полный вектор в виде списка
#         #     st.markdown("**Полный вектор (первые 100 значений):**")
#         #     vector_str = ", ".join([f"{v:.6f}" for v in vector[0][:100]])
#         #     if len(vector[0]) > 100:
#         #         vector_str += f"... (и еще {len(vector[0])-100} значений)"
            
#         #     st.code(vector_str, language="python")
        
#         # with tab3:
#         #     # Детальная информация
#         #     col_info1, col_info2 = st.columns(2)
            
#         #     with col_info1:
#         #         st.markdown("**Статистика вектора:**")
#         #         stats_data = {
#         #             "Метрика": ["Среднее", "Медиана", "Стандартное отклонение", 
#         #                        "Минимум", "Максимум", "Сумма"],
#         #             "Значение": [
#         #                 f"{np.mean(vector[0]):.6f}",
#         #                 f"{np.median(vector[0]):.6f}",
#         #                 f"{np.std(vector[0]):.6f}",
#         #                 f"{np.min(vector[0]):.6f}",
#         #                 f"{np.max(vector[0]):.6f}",
#         #                 f"{np.sum(vector[0]):.6f}"
#         #             ]
#         #         }
#         #         stats_df = pd.DataFrame(stats_data)
#         #         st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
#         #     with col_info2:
#         #         st.markdown("**Исходный текст:**")
#         #         st.info(input_text)
                
#         #         st.markdown("**Информация о векторе:**")
#         #         st.json({
#         #             "dimension": len(vector[0]),
#         #             "shape": vector.shape,
#         #             "dtype": str(vector.dtype),
#         #             "normalized": True,
#         #             "model": "paraphrase-multilingual-MiniLM-L12-v2"
#         #         })
        
#         # # Кнопки для экспорта
#         # st.markdown("---")
#         # st.markdown("### 💾 Экспорт результатов")
        
#         # col_exp1, col_exp2, col_exp3 = st.columns(3)
        
#         # with col_exp1:
#         #     # JSON экспорт
#         #     export_data = {
#         #         "text": input_text,
#         #         "vector": vector[0].tolist(),
#         #         "dimension": len(vector[0]),
#         #         "statistics": {
#         #             "mean": float(np.mean(vector[0])),
#         #             "std": float(np.std(vector[0])),
#         #             "min": float(np.min(vector[0])),
#         #             "max": float(np.max(vector[0]))
#         #         }
#         #     }
            
#         #     json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
#         #     st.download_button(
#         #         label="📥 Скачать JSON",
#         #         data=json_str,
#         #         file_name="vector_result.json",
#         #         mime="application/json"
#         #     )
        
#         # with col_exp2:
#         #     # CSV экспорт
#         #     csv_data = pd.DataFrame({"value": vector[0]})
#         #     csv_str = csv_data.to_csv(index=False)
#         #     st.download_button(
#         #         label="📥 Скачать CSV",
#         #         data=csv_str,
#         #         file_name="vector_values.csv",
#         #         mime="text/csv"
#         #     )
        
#         # with col_exp3:
#         #     # NumPy экспорт
#         #     np_bytes = vector[0].astype(np.float32).tobytes()
#         #     st.download_button(
#         #         label="📥 Скачать .npy",
#         #         data=np_bytes,
#         #         file_name="vector.npy",
#         #         mime="application/octet-stream"
#         #     )

# elif process_btn and not input_text:
#     st.warning("⚠️ Пожалуйста, введите текст для векторизации!")

# # Футер
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: #6b7280; padding: 20px;'>"
#     "Преобразователь текста в векторы • Модель поддерживает кириллицу и другие языки • "
#     "Размерность вектора: 384"
#     "</div>",
#     unsafe_allow_html=True
# )

# # Боковая панель
# with st.sidebar:
#     st.markdown("## ⚙️ Настройки")
    
#     st.markdown("### Параметры модели")
#     model_info = st.expander("Информация о модели", expanded=False)
#     with model_info:
#         st.markdown("""
#         **paraphrase-multilingual-MiniLM-L12-v2**
        
#         - Размерность: 384
#         - Поддерживаемые языки: 50+
#         - Размер модели: ~420 МБ
#         - Использует BERT архитектуру
        
#         **Применение:**
#         - Поиск похожих текстов
#         - Кластеризация
#         - Классификация
#         - Семантический поиск
#         """)
    
#     st.markdown("### Дополнительно")
    
#     if st.button("🔄 Обновить модель", help="Очистить кэш и перезагрузить модель"):
#         st.cache_resource.clear()
#         st.success("Кэш модели очищен! Обновите страницу.")
    
#     st.markdown("---")
    
#     st.markdown("### 📊 Статистика сессии")
#     if 'vectorization_count' not in st.session_state:
#         st.session_state.vectorization_count = 0
    
#     if process_btn and input_text:
#         st.session_state.vectorization_count += 1
    
#     st.metric("Количество векторизаций", st.session_state.vectorization_count)
    
#     st.markdown("---")
#     st.markdown("""
#     ### 🎯 Использование:
#     1. Введите текст в поле слева
#     2. Нажмите "Преобразовать в вектор"
#     3. Изучите результаты
#     4. Экспортируйте данные при необходимости
#     """)

# # Инструкция при первом запуске
# if 'first_run' not in st.session_state:
#     st.session_state.first_run = True
    
# if st.session_state.first_run and not input_text:
#     st.info("💡 **Начните работу**: Введите текст в поле выше и нажмите 'Преобразовать в вектор'")
#     st.session_state.first_run = False