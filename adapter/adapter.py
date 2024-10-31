import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import logging
from engine.plan_adapt_workflow import NarrativeWorkflow
from engine.actor_critic_workflow import ActorCriticWorkflow
from engine.dimensional_critic_actor_engine import DimensionalCriticActorWorkflow
from engine.selective_critic_actor_engine import SelectiveCriticActorWorkflow
from engine.optimizing_critic_actor_engine import OptimizingCriticActorWorkflow
from .save_metadata_adapter import SaveMetadataAdapter, SaveMetadata
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

load_dotenv(override=True)


logger = logging.getLogger('workflow_adapter')

# MongoDB connection
mongo_client = MongoClient(os.getenv('MONGODB_URI'))
db = mongo_client[os.getenv('MONGODB_DB_NAME')]
saves_collection = db[os.getenv('MONGODB_SAVES_COLLECTION')]
metadata_collection = db[os.getenv('MONGODB_METADATA_COLLECTION')]

@dataclass
class StoryState:
    plot: str
    current_scene: str
    scene_history: List[str]
    chat_messages: List[Dict[str, str]]
    timestamp: str
    metadata: Dict[str, Any]
    story_name: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert state to a serializable dictionary."""
        return {
            "plot": self.plot,
            "current_scene": self.current_scene,
            "scene_history": self.scene_history,
            "chat_messages": self.chat_messages,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "story_name": self.story_name
        }
    
    def is_continuation_of(self, other: 'StoryState') -> bool:
        """
        Check if this state is just a continuation of another state.
        Returns True if this state only adds new scenes without modifying existing content.
        """
        if not other:
            return False
            
        # Check if core elements match
        if self.plot != other.plot:
            return False
            
        # Get messages excluding welcome message
        self_messages = self.chat_messages[1:]
        other_messages = other.chat_messages[1:]
        
        # If we have fewer messages, can't be a continuation
        if len(self_messages) <= len(other_messages):
            return False
            
        # Check if all previous messages match exactly
        for self_msg, other_msg in zip(self_messages, other_messages):
            if self_msg != other_msg:
                return False
                
        # If we got here, all previous content matches and we only added new messages
        return True

class WorkflowAdapter:
    WORKFLOW_TYPES = {
        "plan-adapt": NarrativeWorkflow,
        "actor-critic": ActorCriticWorkflow,
        "dimensional-critic": DimensionalCriticActorWorkflow,
        "selective-critic": SelectiveCriticActorWorkflow,
        "optimizing-critic": OptimizingCriticActorWorkflow
    }

    def __init__(self, save_dir: str = "saves"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.current_state: Optional[StoryState] = None
        self.current_save_path: Optional[str] = None
        self.current_save_id: Optional[str] = None
        self.metadata_adapter = SaveMetadataAdapter(save_dir)
        logger.info("WorkflowAdapter initialized with save directory: %s", save_dir)

    def _get_workflow_class(self, config: Dict[str, Any]) -> Any:
        """Get the appropriate workflow class based on config."""
        workflow_type = config.get("workflow_type", "plan-adapt")
        return self.WORKFLOW_TYPES.get(workflow_type, NarrativeWorkflow)

    def _generate_save_path(self) -> str:
        """Generate a unique save file path with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return os.path.join(self.save_dir, f"story_state_{timestamp}.json")

    def _extract_narrative_pairs(self, chat_messages: List[Dict[str, str]], max_scenes: int) -> Tuple[List[str], List[str]]:
        """
        Extract user actions and scene responses as pairs, limited to max_scenes.
        Returns (actions, scenes) tuple.
        """
        # Skip only the welcome message
        messages = [msg for msg in chat_messages[1:]]  # Keep initial scene
        
        # Group into pairs of (user_action, scene_response)
        pairs = []
        current_pair = []
        
        for msg in messages:
            current_pair.append(msg["content"])
            if len(current_pair) == 2:  # We have a complete pair
                pairs.append(tuple(current_pair))
                current_pair = []
        
        # Take only the last max_scenes pairs
        pairs = pairs[-max_scenes:] if len(pairs) > max_scenes else pairs
        
        # Unzip pairs into separate lists
        actions, scenes = zip(*pairs) if pairs else ([], [])
        return list(actions), list(scenes)

    async def save_state(self, state: Optional[StoryState] = None, workflow_config: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Save the current or provided story state to both disk and MongoDB.
        If this is a continuation of a loaded save, it will update that save.
        If this is a regeneration, it will create a new save.
        """
        state = state or self.current_state
        if not state:
            raise ValueError("No state to save")

        try:
            # Generate metadata using LLM
            save_metadata = await self.metadata_adapter.generate_metadata(
                plot=state.plot,
                chat_messages=state.chat_messages,
                workflow_config=workflow_config
            )

            # Check if this is a regeneration
            is_regeneration = state.metadata.get("regenerated", False)

            # If this is a regeneration, always create a new save
            # Otherwise, use the current save path/id if they exist
            if is_regeneration:
                save_path = self._generate_save_path()
                save_id = None
            else:
                save_path = self.current_save_path or self._generate_save_path()
                save_id = self.current_save_id
            
            # Convert state to serializable dictionary
            state_dict = state.to_dict()
            
            # Save to local file
            with open(save_path, 'w') as f:
                json.dump(state_dict, f, indent=2)

            # Save metadata to separate file
            self.metadata_adapter.save_metadata(save_path, save_metadata)

            # Save to MongoDB - separate collections for state and metadata
            if save_id:  # Update existing documents
                saves_collection.update_one(
                    {"_id": ObjectId(save_id)},
                    {"$set": state_dict}
                )
                metadata_collection.update_one(
                    {"save_id": save_id},
                    {"$set": save_metadata.to_dict()}
                )
            else:  # Insert new documents
                result = saves_collection.insert_one(state_dict)
                save_id = str(result.inserted_id)
                metadata_dict = save_metadata.to_dict()
                metadata_dict['save_id'] = save_id
                metadata_collection.insert_one(metadata_dict)

            # Update current save path and id
            self.current_save_path = save_path
            self.current_save_id = save_id
            
            action = "created new" if is_regeneration else "updated"
            logger.info(f"State {action} successfully at: {save_path} and MongoDB ID: {save_id}")
            return save_path, save_id
            
        except Exception as e:
            logger.error("Failed to save state: %s", str(e))
            raise

    def load_state(self, identifier: str) -> StoryState:
        """
        Load a story state from disk or MongoDB.
        If not found in one, it will try the other.
        """
        local_state = None
        mongo_state = None
        metadata = None

        # Try to load from local file
        local_path = os.path.join(self.save_dir, identifier)
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r') as f:
                    state_dict = json.load(f)
                    # Remove '_id' if present (from MongoDB saves)
                    state_dict.pop('_id', None)
                    local_state = StoryState(**state_dict)
                # Load local metadata
                metadata = self.metadata_adapter.load_metadata(local_path)
                if metadata:
                    local_state.metadata.update(metadata.to_dict())
                logger.info(f"State loaded from local file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to load state from local file: {str(e)}")

        # Try to load from MongoDB
        try:
            # Try to parse as ObjectId first
            try:
                obj_id = ObjectId(identifier)
                mongo_doc = saves_collection.find_one({"_id": obj_id})
                if mongo_doc:
                    # Convert _id to string and remove it from the document
                    mongo_doc['_id'] = str(mongo_doc['_id'])
                    mongo_id = mongo_doc.pop('_id')
                    mongo_state = StoryState(**mongo_doc)
                    # Load metadata from separate collection
                    metadata_doc = metadata_collection.find_one({"save_id": mongo_id})
                    if metadata_doc:
                        mongo_state.metadata.update(metadata_doc)
                    logger.info(f"State loaded from MongoDB with ID: {identifier}")
            except Exception as e:
                logger.warning(f"Failed to parse MongoDB ID: {str(e)}")
        except Exception as e:
            logger.warning(f"Failed to load state from MongoDB: {str(e)}")

        # Determine which state to use
        if local_state and mongo_state:
            # Use the most recent one
            state = local_state if local_state.timestamp >= mongo_state.timestamp else mongo_state
        elif local_state:
            state = local_state
        elif mongo_state:
            state = mongo_state
        else:
            raise ValueError(f"Failed to load state with identifier: {identifier}")

        self.current_state = state
        self.current_save_path = local_path if local_state else None
        self.current_save_id = identifier if mongo_state else None

        return state

    def list_saves(self) -> List[Dict[str, str]]:
        """List all available save files with their metadata from both local directory and MongoDB."""
        try:
            saves = {}
            
            # List local saves
            for f in os.listdir(self.save_dir):
                if f.startswith("story_state_") and f.endswith(".json"):
                    save_path = os.path.join(self.save_dir, f)
                    timestamp = f.replace("story_state_", "").replace(".json", "")
                    display_text = self.metadata_adapter.format_save_display(save_path)
                    saves[timestamp] = {
                        "path": f,
                        "display": display_text,
                        "timestamp": timestamp,
                        "source": "local"
                    }

            # List MongoDB saves
            mongo_saves = saves_collection.find({}, {"_id": 1, "timestamp": 1})
            for save in mongo_saves:
                timestamp = save["timestamp"]
                mongo_id = str(save["_id"])
                
                if timestamp in saves:
                    # Save exists in both local and MongoDB
                    saves[timestamp]["source"] = "both"
                    saves[timestamp]["mongo_id"] = mongo_id
                else:
                    # Save exists only in MongoDB
                    metadata = metadata_collection.find_one({"save_id": mongo_id})
                    display_text = f"MongoDB save from {timestamp}"
                    if metadata:
                        display_text = self.metadata_adapter.format_mongo_save_display(metadata)
                    
                    saves[timestamp] = {
                        "mongo_id": mongo_id,
                        "display": display_text,
                        "timestamp": timestamp,
                        "source": "mongo"
                    }

            # Sort by timestamp
            sorted_saves = sorted(saves.values(), key=lambda x: x["timestamp"], reverse=True)

            # Format the final list
            return [{
                "id": save.get("path") or save.get("mongo_id"),
                "display": save["display"],
                "source": save["source"]
            } for save in sorted_saves]

        except Exception as e:
            logger.error("Failed to list saves: %s", str(e))
            raise

    def rollback_to_state(self, identifier: str) -> StoryState:
        """Roll back to a previous story state."""
        return self.load_state(identifier)

    async def generate_next_state(self, 
                                user_action: str, 
                                chat_messages: List[Dict[str, str]],
                                max_scenes: int,
                                workflow_config: Optional[Dict[str, Any]] = None,
                                timeout: int = 120) -> StoryState:
        """Generate the next story state based on user action."""
        if not self.current_state:
            raise ValueError("No current state to generate from")

        try:
            # Create workflow instance with provided config
            WorkflowClass = self._get_workflow_class(workflow_config or {})
            workflow = WorkflowClass(
                config=workflow_config or {},
                timeout=timeout
            )

            # Extract narrative pairs limited to max_scenes
            actions, scenes = self._extract_narrative_pairs(chat_messages, max_scenes)
            logger.info(f"Using {len(scenes)} scene pairs for context generation")

            # Combine actions and scenes alternately for complete context
            narrative_context = []
            for action, scene in zip(actions, scenes):
                narrative_context.extend([action, scene])

            # Run workflow with narrative context
            result = await workflow.run(
                plot=self.current_state.plot,
                current_scene=self.current_state.current_scene,
                user_action=user_action,
                scene_history=narrative_context
            )

            # Handle string result (error case)
            if isinstance(result, str):
                raise ValueError(result)
            
            # Create new state from workflow result
            new_state = StoryState(
                plot=self.current_state.plot,
                current_scene=result["narrative"],
                scene_history=scenes + [self.current_state.current_scene],
                chat_messages=chat_messages,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "original_vision": result["original_vision"],
                    "user_action": user_action,
                    "model_config": workflow_config
                },
                story_name=self.current_state.story_name
            )
            
            self.current_state = new_state
            logger.info("Generated new state successfully")
            return new_state
            
        except Exception as e:
            logger.error("Failed to generate next state: %s", str(e))
            raise

    async def regenerate_current_state(self, 
                                     chat_messages: List[Dict[str, str]],
                                     max_scenes: int,
                                     workflow_config: Optional[Dict[str, Any]] = None,
                                     timeout: int = 120) -> StoryState:
        """Regenerate the current state with potentially different configuration."""
        if not self.current_state:
            raise ValueError("No current state to regenerate")

        try:
            # Get the last user action from metadata
            user_action = self.current_state.metadata.get("user_action")
            if not user_action:
                raise ValueError("No user action found in current state metadata")

            # Extract narrative pairs limited to max_scenes
            actions, scenes = self._extract_narrative_pairs(chat_messages, max_scenes)
            logger.info(f"Using {len(scenes)} scene pairs for context regeneration")

            # Get the previous scene
            prev_scene = scenes[-1] if scenes else self.current_state.current_scene

            # Combine actions and scenes alternately for complete context
            narrative_context = []
            for action, scene in zip(actions[:-1], scenes[:-1]):  # Exclude last pair
                narrative_context.extend([action, scene])

            # Create new workflow instance with provided config
            WorkflowClass = self._get_workflow_class(workflow_config or {})
            workflow = WorkflowClass(
                config=workflow_config or {},
                timeout=timeout
            )

            # Run workflow with narrative context
            result = await workflow.run(
                plot=self.current_state.plot,
                current_scene=prev_scene,
                user_action=user_action,
                scene_history=narrative_context
            )

            # Handle string result (error case)
            if isinstance(result, str):
                raise ValueError(result)

            # Create new state with regenerated content
            new_state = StoryState(
                plot=self.current_state.plot,
                current_scene=result["narrative"],
                scene_history=scenes,
                chat_messages=chat_messages,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                metadata={
                    "original_vision": result["original_vision"],
                    "user_action": user_action,
                    "model_config": workflow_config,
                    "regenerated": True  # Mark this as a regeneration
                },
                story_name=self.current_state.story_name
            )
            
            self.current_state = new_state
            logger.info("Regenerated state successfully")
            return new_state
            
        except Exception as e:
            logger.error("Failed to regenerate state: %s", str(e))
            raise

    def create_initial_state(self, 
                           plot: str, 
                           current_scene: str,
                           chat_messages: List[Dict[str, str]],
                           scene_history: Optional[List[str]] = None) -> StoryState:
        """Create and set the initial story state."""
        state = StoryState(
            plot=plot,
            current_scene=current_scene,
            scene_history=scene_history or [],
            chat_messages=chat_messages,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata={"initial_state": True},
            story_name=None
        )
        self.current_state = state
        logger.info("Created initial state")
        return state
