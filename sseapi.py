

import json
import requests
import asyncio
import aiohttp

from typing import List, Dict, Any, Callable

#Ты чат бот ассистент, отвечающий на вопросы по учебным материалам пользователя.как в 1с сделать запрос на остатки товаров на складе
SYSTEM_MESSAGE="""
Ты — специализированный ассистент по работе с учебными материалами. Твоя задача — точно и понятно отвечать на вопросы пользователя, используя предоставленные фрагменты конспектов. Тебе надо указать номер страницы
Лекционные материалы приведены ниже в виде абзацев json. Укажи номера страниц найденных фрамнетов "page" и наименованике документа "name_dock". 
Если фрагменты не совсем отвечают на вопрос пользователя, указать на страницы тех фрагментов которые больше подходят теме вопроса и коротко ответить самостоятельно.
"""

class BotMessageModel:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {"role": self.role, "content": self.content}



class SSE_Bot:
    def __init__(self,url:str="http://localhost:11434/",model="deepseek-v3.1:671b-cloud"): 
        #https://doubtlessly-fitting-ocelot.cloudpub.ru/
        #streamlit run PythonAppSearchText.py
        self.base_url=url
        self.model_name=model


     
    
    def procces_message_line(
        self,
        line: str,
    )->str:
       # print(line)

        if not line:
          #  print(line,"[none]")
            return "[none]"
        if(line.strip() == "[null]" or 
            line.strip() == "[DONE]" or 
            line.strip() == "null" or
            "[null]" in line.strip()):
          #  print(line,"[null]")
            return "[null]"

        json_data = line
        if line.startswith("data: "):
         #   print("вырезаем line.startswith(data: )  line[6:].strip()data: ")
            json_data = line[6:].strip()
            
        # Пропускаем пустые данные после извлечения
        
        stream_response = json.loads(json_data)
                
        # Проверяем структуру ответа Ollama
        if "message" in stream_response and "content" in stream_response["message"]:
            content = stream_response["message"]["content"]
            if content:
         #       print("content line=",content)
                return content
                
        # Проверяем флаг завершения
        if stream_response.get("done", False):
         #   print(line,"[done]")
            return "[done]"
        return "[none]"


    async def request(self,
        promt:str,
        documents:str,
        system_message: str=SYSTEM_MESSAGE,
        num_predict: int = 1000)->aiohttp.ClientResponse:
        try:

            # можно расширить до истории чата
            context = [
                BotMessageModel(role="system", content=system_message).to_dict(),
                BotMessageModel(role="system", content=f"Найденные фрагменты в лекциях по запросу пользователя в формате json: {documents}").to_dict(),
                BotMessageModel(role="user", content=promt).to_dict()]

            print(context)
            request_body = {
                "model": self.model_name,
                "messages": context,
                "stream": True,
                "options": {
                    "num_predict": num_predict
                }
            }

            url = f"{self.base_url.rstrip('/')}/api/chat"
            print("запрос нейросети")
            session= aiohttp.ClientSession()
            try:
                response=await session.post(
                    url,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json"#,"Authorize":""
                    }
                ) 
                
                print("response получен")
                response.raise_for_status()
                
                print("stream получен")
                
                # Добавляем небольшую задержку как в C# коде
                await asyncio.sleep(1)
                return response, session
            except aiohttp.ClientError as e:
                print(f"Ошибка при выполнении запроса: {e}")
                raise
            # response=requests.post(url,
            #             json=request_body,
            #             stream=True,
            #             headers={"Content-Type": "application/json"},
            #             timeout=None)
            # print("ответ получен")
            # response.raise_for_status()
            # await asyncio.sleep(1)
            # return response
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при выполнении запроса: {e}")
            raise
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            raise
        return None



    async def send_streaming_request_async(
        self,
        system_message: str,
        messages_chat: List[BotMessageModel],
        on_chunk_received: Callable[[str], None],
        num_predict: int = 1000
    ):
        """
        Отправляет потоковый запрос к Ollama API
    
        Args:
            system_message: Системное сообщение
            messages_chat: История сообщений чата
            base_url: Базовый URL API (например, "http://localhost:11434")
            model_name: Имя модели Ollama
            on_chunk_received: Callback функция для обработки полученных chunk'ов
            cancellation_token: Токен отмены (для совместимости)
            num_predict: Максимальное количество токенов для генерации
        """
        context = [BotMessageModel(role="system", content=system_message).to_dict()]
        context.extend([msg.to_dict() for msg in messages_chat])
        request_body = {
            "model": self.model_name,
            "messages": context,
            "stream": True,
            "options": {
                "num_predict": num_predict
            }
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"

        try:
            response=requests.post(url,
                                   json=request_body,
                                   stream=True,
                                   headers={"Content-Type": "application/json"},
                                   timeout=None)
            response.raise_for_status()
            await asyncio.sleep(1)
            for line in  response.iter_lines():

                if not line:
                    continue
                if  (line.strip() == "[null]" or 
                    line.strip() == "[DONE]" or 
                    line.strip() == "null" or
                    "[null]" in line.strip()):
                    break
                json_data = line
                if line.startswith("data: "):
                    json_data = line[6:].strip()
            
                # Пропускаем пустые данные после извлечения
                if not json_data or json_data in ["[null]", "[DONE]", "null"]:
                    continue

                try:
                    # Десериализуем JSON
                    stream_response = json.loads(json_data)
                
                    # Проверяем структуру ответа Ollama
                    if "message" in stream_response and "content" in stream_response["message"]:
                        content = stream_response["message"]["content"]
                        if content:
                            # Вызываем callback с полученным chunk'ом
                            await on_chunk_received(content)
                
                    # Проверяем флаг завершения
                    if stream_response.get("done", False):
                        print("Stream completed (done=true)")
                        break
                except json.JSONDecodeError as e:
                    print(f"Ошибка десериализации JSON: {e}, data: {json_data}")
                except Exception as e:
                    print(f"Ошибка обработки chunk: {e}")
                await asyncio.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при выполнении запроса: {e}")
            raise
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            raise





# # Синхронный пример
# async def handle_chunk(chunk: str):
#     print(chunk, end="", flush=True)

# # Подготовка данных
# system_msg = "Ты полезный ассистент"
# chat_history = [
#     BotMessageModel("user", "Привет, как дела?"),
#     BotMessageModel("assistant", "Привет! У меня все хорошо, спасибо!")
# ]

# # Вызов (в асинхронном контексте)
# await send_streaming_request(
#     system_message=system_msg,
#     messages_chat=chat_history,
#     base_url="http://localhost:11434",
#     model_name="llama3.2",
#     on_chunk_received=handle_chunk
# )