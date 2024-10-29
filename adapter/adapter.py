import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from engine.plan_adapt_workflow import NarrativeWorkflow
from engine.actor_critic_workflow import ActorCriticWorkflow
from engine.dimensional_critic_actor_engine import DimensionalCriticActorWorkflow
from engine.selective_critic_actor_engine import SelectiveCriticActorWorkflow
from engine.optimizing_critic_actor_engine import OptimizingCriticActorWorkflow
from .save_metadata_adapter import SaveMetadataAdapter, SaveMetadata
from database.mongo_client import MongoClient

logger = logging.getLogger('workflow_adapter')

@dataclass
class StoryState:
    plot: str
    current_scene: str
    scene_history: List[str]
    chat_messages: List[Dict[str, str]]
    timestamp: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert state to a serializable dictionary."""
        return {
            "plot": self.plot,
            "current_scene": self.current_scene,
            "scene_history": self.scene_history,
            "chat_messages": self.chat_messages,
            "timestamp": self.timestamp,
            "metadata": self.metadata
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

    def __init__(self, mongo_client: MongoClient):
        self.current_state: Optional[StoryState] = None
        self.current_save_id: Optional[str] = None
        self.metadata_adapter = SaveMetadataAdapter(mongo_client)
        logger.info("WorkflowAdapter initialized with MongoDB client")

    def _get_workflow_class(self, config: Dict[str, Any]) -> Any:
        """Get the appropriate workflow class based on config."""
        workflow_type = config.get("workflow_type", "plan-adapt")
        return self.WORKFLOW_TYPES.get(workflow_type, NarrativeWorkflow)

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

    async def save_state(self, state: Optional[StoryState] = None, workflow_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Save the current or provided story state to MongoDB.
        If the state is a continuation of the current save, it will update it.
        Otherwise, it creates a new save document.
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

            # Combine state and metadata
            save_data = {**state.to_dict(), **save_metadata.to_dict()}

            if self.current_save_id:
                # Update existing save
                await self.metadata_adapter.db_client.update_metadata(self.current_save_id, save_data)
                save_id = self.current_save_id
                logger.info(f"State updated successfully with ID: {save_id}")
            else:
                # Create new save
                save_id = await self.metadata_adapter.db_client.save_metadata(save_data)
                self.current_save_id = save_id
                logger.info(f"New state saved successfully with ID: {save_id}")

            return save_id

        except Exception as e:
            logger.error("Failed to save state: %s", str(e))
            raise

    async def load_state(self, save_id: str) -> StoryState:
        """Load a story state from MongoDB."""
        try:
            data = await self.metadata_adapter.db_client.load_metadata(save_id)
            if not data:
                raise ValueError(f"No state found with ID: {save_id}")

            # Remove MongoDB _id field before creating StoryState
            if '_id' in data:
                del data['_id']

            state = StoryState(
                plot=data["plot"],
                current_scene=data["current_scene"],
                scene_history=data["scene_history"],
                chat_messages=data["chat_messages"],
                timestamp=data["timestamp"],
                metadata=data.get("metadata", {})  # Use get() with default empty dict
            )
            self.current_state = state
            self.current_save_id = save_id
            logger.info(f"State loaded successfully from ID: {save_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            raise

    async def list_saves(self) -> List[Dict[str, Any]]:
        """List all available saves with their metadata."""
        try:
            saves = await self.metadata_adapter.db_client.list_saves()
            return [{"id": str(save["_id"]), "name": save["story_name"], "timestamp": save["timestamp"]} for save in saves]
        except Exception as e:
            logger.error(f"Failed to list saves: {str(e)}")
            raise

    async def rollback_to_state(self, save_id: str) -> StoryState:
        """Roll back to a previous story state."""
        return await self.load_state(save_id)

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
                }
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
                    "regenerated": True
                }
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
            metadata={"initial_state": True}
        )
        self.current_state = state
        logger.info("Created initial state")
        return state
