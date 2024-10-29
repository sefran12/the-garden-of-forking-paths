import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from llama_index.core.llms.llm import LLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from database.mongo_client import MongoClient

logger = logging.getLogger('save_metadata_adapter')

@dataclass
class SaveMetadata:
    story_name: str
    overall_summary: str
    latest_summary: str
    timestamp: str
    save_id: Optional[str] = None

    def to_dict(self) -> Dict:
        data = {
            "story_name": self.story_name,
            "overall_summary": self.overall_summary,
            "latest_summary": self.latest_summary,
            "timestamp": self.timestamp
        }
        return data

class SaveMetadataAdapter:
    def __init__(self, db_client: MongoClient, save_dir: str = "saves"):
        self.save_dir = save_dir
        self.db_client = db_client
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._llm: Optional[LLM] = None

    async def _initialize_llm(self, config: Dict[str, Any]) -> LLM:
        """Initialize LLM based on provider and model configuration."""
        if self._llm is None:
            provider = config.get("provider", "openai")
            model = config.get("model", "gpt-4-mini")
            
            logger.info(f"Initializing LLM with provider: {provider}, model: {model}")
            
            try:
                if provider == "ollama":
                    self._llm = Ollama(model=model, temperature=0.7)
                elif provider == "openai":
                    self._llm = OpenAI(model=model, temperature=0.7)
                elif provider == "anthropic":
                    self._llm = Anthropic(model=model, temperature=0.7)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                logger.info("LLM initialization successful")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        
        return self._llm

    async def generate_metadata(self, 
                              plot: str,
                              chat_messages: List[Dict[str, str]], 
                              workflow_config: Optional[Dict[str, Any]] = None) -> SaveMetadata:
        """Generate metadata for the current story state using LLM."""
        try:
            llm = await self._initialize_llm(workflow_config or {})

            # Extract scene pairs (excluding welcome message)
            scene_pairs = []
            messages = chat_messages[1:]  # Skip welcome message
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    action = messages[i]["content"]
                    scene = messages[i+1]["content"]
                    scene_pairs.append((action, scene))

            # Generate story name
            name_prompt = f"""
            Based on this story:
            Setting: {plot}

            Recent events:
            {self._format_scenes(scene_pairs[-5:] if len(scene_pairs) > 5 else scene_pairs)}

            Create a clear, descriptive title (max 50 characters) that captures the main elements of the story.
            Focus on concrete details like location, characters, or central conflict.
            Start directly with the title - do not include any introductory phrases.
            """
            
            story_name = (await llm.acomplete(name_prompt)).text.strip()

            # Generate overall summary
            overall_prompt = f"""
            Summarize this story:
            Setting: {plot}

            Events in order:
            {self._format_scenes(scene_pairs[:10])}  # Limit to first 10 scenes

            Write a 200-word factual summary focusing on:
            - Who are the main characters and what are their roles
            - Where does the story take place (specific locations)
            - What key events have happened
            - What is the current situation

            Important instructions:
            - Start directly with the summary - do not include phrases like "Here's a summary" or "The story is about"
            - Keep the summary focused on concrete events and facts
            - Avoid philosophical interpretations or thematic analysis
            - Write in present tense
            """
            
            overall_summary = (await llm.acomplete(overall_prompt)).text.strip()

            # Generate latest summary
            latest_prompt = f"""
            Summarize these recent events:
            {self._format_scenes(scene_pairs[-3:] if len(scene_pairs) > 3 else scene_pairs)}

            Write a 100-word factual summary that covers:
            - What specifically happened in these scenes
            - Who was involved
            - Where these events took place
            - What is the immediate situation now

            Important instructions:
            - Start directly with the events - do not include phrases like "In these scenes" or "These events show"
            - Focus only on describing the actual events and current state
            - Avoid speculation about implications or deeper meaning
            - Write in present tense
            """
            
            latest_summary = (await llm.acomplete(latest_prompt)).text.strip()

            return SaveMetadata(
                story_name=story_name,
                overall_summary=overall_summary,
                latest_summary=latest_summary,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

        except Exception as e:
            logger.error(f"Failed to generate save metadata: {str(e)}")
            raise

    def _format_scenes(self, scene_pairs: List[Tuple[str, str]]) -> str:
        """Format scene pairs for prompt context."""
        formatted = []
        for i, (action, scene) in enumerate(scene_pairs, 1):
            formatted.extend([
                f"Scene {i}:",
                f"Action: {action}",
                f"Result: {scene}",
                ""
            ])
        return "\n".join(formatted)

    async def save_metadata(self, save_path: str, metadata: SaveMetadata):
        """Save metadata to both file system and MongoDB."""
        try:
            # Save to MongoDB
            metadata_dict = metadata.to_dict()
            metadata_dict['save_path'] = save_path
            save_id = await self.db_client.save_metadata(metadata_dict)
            metadata.save_id = save_id

            # Save to file system
            metadata_path = self._get_metadata_path(save_path)
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise

    async def load_metadata(self, save_id: str) -> Optional[SaveMetadata]:
        """Load metadata from MongoDB."""
        try:
            mongo_data = await self.db_client.load_metadata(save_id)
            if mongo_data:
                # Extract metadata fields from root level
                metadata_fields = {
                    'story_name': mongo_data.get('story_name'),
                    'overall_summary': mongo_data.get('overall_summary'),
                    'latest_summary': mongo_data.get('latest_summary'),
                    'timestamp': mongo_data.get('timestamp'),
                    'save_id': str(mongo_data['_id'])
                }
                # Only create metadata if all required fields are present
                if all(metadata_fields.values()):
                    return SaveMetadata(**metadata_fields)
            return None
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return None

    def _get_metadata_path(self, save_path: str) -> str:
        """Get the path for the metadata file associated with a save file."""
        base, _ = os.path.splitext(save_path)
        return f"{base}_metadata.json"

    async def format_save_display(self, save_id: str) -> str:
        """Format save information for display in UI."""
        try:
            metadata = await self.load_metadata(save_id)
            if metadata:
                return (f"Save ID: {save_id}\n"
                       f"Story: {metadata.story_name}\n"
                       f"Last updated: {metadata.timestamp}\n\n"
                       f"Latest events: {metadata.latest_summary}")
            return f"Save ID: {save_id}"
        except Exception as e:
            logger.error(f"Failed to format save display: {str(e)}")
            return f"Save ID: {save_id}"
