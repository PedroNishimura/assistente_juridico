# Assistente de Escritório

Assistente conversacional em português para apoiar tarefas de escritório, como redação, revisão, síntese e organização. Também permite enviar um PDF e fazer perguntas com respostas fundamentadas em trechos recuperados do documento (RAG).

## Funcionalidades

- Conversa com histórico durante a sessão.
- Consulta opcional a PDFs, com indicação de arquivo e página.
- Recuperação semântica com embeddings multilíngues e busca MMR para trazer trechos relevantes e variados.
- Chave de API carregada de variável de ambiente ou arquivo `.env`, sem campo de chave na interface.
- Modelo de embeddings mantido em cache entre interações para reduzir espera.

## Requisitos

- Python 3.10 ou superior.
- Uma chave de API da Groq.
- Conexão com a internet na primeira execução para baixar o modelo de embeddings e para conversar com a API.

## Instalação

```bash
git clone <URL_DO_REPOSITORIO>
cd assistente_juridico
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Instale as dependências e configure a chave:

```bash
python -m pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha `GROQ_API_KEY`. O arquivo `.env` é ignorado pelo Git. Também é possível definir `GROQ_API_KEY` diretamente no ambiente. `GROQ_MODEL` é opcional e assume `openai/gpt-oss-20b`.

## Execução

```bash
streamlit run main.py
```

Abra o endereço local exibido pelo Streamlit. Envie um PDF pela barra lateral para habilitar a consulta documental. PDFs com texto selecionável funcionam diretamente; documentos digitalizados precisam passar por OCR antes do envio.

## Docker

É necessário ter Docker Desktop com o mecanismo Linux ativo. Se ainda não houver um `.env`, copie `.env.example` e preencha `GROQ_API_KEY` (no PowerShell, use `Copy-Item .env.example .env`; no macOS/Linux, `cp .env.example .env`). Em seguida, rode:

```bash
docker compose up --build -d
```

No primeiro início, o container baixa o modelo multilíngue de embeddings; as próximas execuções reutilizam o volume `modelo_huggingface`. Acesse [http://localhost:8501](http://localhost:8501). Para acompanhar os logs, use `docker compose logs -f`; para encerrar, use `docker compose down`.

A imagem usa Python 3.12 e PyTorch para CPU, roda como usuário sem privilégios e não copia o `.env` para dentro dela. Não é necessária uma placa NVIDIA. Para guardar outro segredo fora do arquivo, remova `env_file` de `compose.yaml` e forneça `GROQ_API_KEY` por um gerenciador de segredos ou pelo ambiente do container.

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes exercitam divisão de documentos, metadados de página e recuperação vetorial sem chamadas à Groq ou download de modelo externo.

## Estrutura

```text
assistente_juridico/
├── assistente/
│   ├── assistant.py  # comportamento e integração com o modelo de conversa
│   ├── config.py     # configuração e variáveis de ambiente
│   ├── rag.py        # leitura, segmentação, indexação e recuperação de PDFs
│   └── ui.py         # interface Streamlit e estado da sessão
├── tests/            # testes locais sem serviços externos
├── Dockerfile        # imagem de execução
├── compose.yaml      # serviço local e cache persistente do modelo
├── main.py           # ponto de entrada
└── requirements.txt
```

## Privacidade e uso

O índice vetorial é mantido em memória na sessão. O texto do PDF recuperado e as mensagens da conversa são enviados ao provedor de modelo para gerar respostas. Não envie documentos que você não esteja autorizado a compartilhar. O assistente oferece apoio informativo e de redação; não substitui aconselhamento profissional.
