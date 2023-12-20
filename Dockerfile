# Используйте официальный образ Python 3.8
FROM python:3.11.6

# Установите зависимости
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
ADD ./src /app/src

# Скопируйте все файлы в образ
COPY . /app

# Укажите рабочую директорию внутри /app/src
WORKDIR /app/src

# Укажите команду для запуска вашего приложения
CMD ["python", "main.py"]
