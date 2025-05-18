# Используем официальный образ Python в качестве базового
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и main.py в контейнер
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update && apt-get install -y curl

# Открываем порт, на котором будет работать приложение
EXPOSE 8510

# Команда для запуска приложения
CMD ["streamlit", "run", "main.py", "--server.port", "8510"]
