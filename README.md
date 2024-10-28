# The Garden of Forking Paths

An interactive narrative application using various language models.

## Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OLLAMA_HOST`: URL for Ollama if using local models

## Docker Setup

1. Build the container:
```bash
docker-compose build
```

2. Run the application:
```bash
docker-compose up
```

The application will be available at http://localhost:8000

## Development

For local development without Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
shiny run
