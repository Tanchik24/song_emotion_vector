FROM python:3.9

WORKDIR /app

# Копирование исходного кода проекта
COPY . .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Указание команды для запуска сервера приложения
CMD ["python3", "main.py"]