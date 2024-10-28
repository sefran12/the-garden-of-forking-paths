# The Garden of Forking Paths

An interactive narrative experience powered by AI.

## Running the Application

You can run the application with different language settings using the following command:

```bash
python run.py --lang [language_code]
```

Available language codes:
- `en` - English (default)
- `it` - Italian

Additional options:
- `--host` - Host to run the server on (default: 127.0.0.1)
- `--port` - Port to run the server on (default: 8000)

Examples:
```bash
# Run with English (default)
python run.py

# Run with Italian
python run.py --lang it

# Run on a specific host and port
python run.py --lang en --host 0.0.0.0 --port 8080
```

## Project Structure

- `app.py` - Main Shiny application
- `run.py` - Application launcher with language selection
- `resource_manager.py` - Manages text resources and translations
- `resources.json` - Contains all text content in different languages
- `engine/` - Core narrative generation logic
- `adapter/` - Interface adapters for the narrative engine
- `saves/` - Directory for saved game states

## Development

To add support for a new language:
1. Add the language content to `resources.json` using the appropriate language code
2. Ensure all text resources are properly translated
3. The new language will automatically be available through the `--lang` parameter

## Requirements

See `requirements.txt` for Python dependencies.
