FROM python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

EXPOSE 8000

CMD ["panel", "serve", "--port", "8000", "app.py", "--address", "0.0.0.0", "--allow-websocket-origin", "*"]