import json
import os
from typing import Dict, Optional

class ResourceManager:
    _instance = None
    _resources: Dict = {}
    _current_language: str = "en"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._resources:
            self.load_resources()
    
    def load_resources(self, language: str = "en") -> None:
        """Load resources from the JSON file."""
        try:
            with open("resources.json", "r", encoding="utf-8") as f:
                self._resources = json.load(f)
            self._current_language = language
        except FileNotFoundError:
            raise FileNotFoundError("resources.json not found. Please ensure it exists in the project root.")
    
    def get_text(self, key: str, default: Optional[str] = None) -> str:
        """
        Get text resource by key for current language.
        Falls back to English if text not found in current language.
        """
        try:
            # Try to get text in current language
            return self._resources[self._current_language][key]
        except KeyError:
            try:
                # Fall back to English if text not found in current language
                if self._current_language != "en":
                    return self._resources["en"][key]
                # If English text not found and default provided, return default
                if default is not None:
                    return default
                # Otherwise raise error
                raise KeyError(f"Resource key '{key}' not found in any language")
            except KeyError:
                if default is not None:
                    return default
                raise KeyError(f"Resource key '{key}' not found in any language")
    
    def set_language(self, language: str) -> None:
        """Set the current language."""
        if language not in self._resources:
            raise ValueError(f"Language '{language}' not supported")
        self._current_language = language
    
    def get_current_language(self) -> str:
        """Get the current language code."""
        return self._current_language
    
    def get_available_languages(self) -> list:
        """Get list of available language codes."""
        return list(self._resources.keys())
