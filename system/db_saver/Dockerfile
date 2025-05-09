# Используем официальный образ Python в качестве базового
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы requirements.txt и main.py в контейнер
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Открываем порт, на котором будет работать приложение
EXPOSE 8501

# Команда для запуска приложения
CMD ["python", "main.py"]
