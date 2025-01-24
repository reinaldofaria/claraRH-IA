# Usar uma imagem base com Python
FROM python:3.13.1-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Atualizar o sistema, instalar dependências essenciais e limpar cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Atualizar o pip
RUN pip install --upgrade pip

# Copiar apenas o arquivo de dependências
COPY requirements.txt .

# Instalar as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os outros arquivos para o container
COPY . .

# Definir o comando para rodar o app Streamlit
CMD ["streamlit", "run", "app.py"]

