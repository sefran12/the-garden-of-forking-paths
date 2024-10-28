import argparse
import logging
from shiny import App
from resource_manager import ResourceManager

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('narrative_launcher')

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Launch Interactive Narrative with language selection')
    parser.add_argument(
        '--lang', 
        type=str, 
        default='en',
        help='Language code (e.g., en, it)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to run the server on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run the server on'
    )

    args = parser.parse_args()

    try:
        # Initialize ResourceManager with selected language
        resource_manager = ResourceManager()
        available_languages = resource_manager.get_available_languages()
        
        if args.lang not in available_languages:
            logger.error(f"Language '{args.lang}' not available. Available languages: {', '.join(available_languages)}")
            return
        
        resource_manager.set_language(args.lang)
        logger.info(f"Set language to: {args.lang}")
        
        # Import app module after setting language
        import app
        
        # Run the Shiny app
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.app.run(host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()
