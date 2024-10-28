import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, ClassVar
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step,
    Event,
    retry_policy
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms.llm import LLM
from resource_manager import ResourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('workflow_engine')

class StoryContext:
    def __init__(self, plot: str, current_scene: str, scene_history: List[str] = None):
        self.plot = plot
        self.current_scene = current_scene
        self.scene_history = scene_history or []
        self.narrative_threads = {}

class PlanningEvent(Event):
    context: StoryContext
    narrative_vision: str

class UserInputEvent(Event):
    context: StoryContext
    user_action: str
    narrative_vision: str

class NarrativeResponseEvent(Event):
    narrative: str
    original_vision: str
    
class NarrativeWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}
    _resource_manager: ResourceManager = ResourceManager()

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        NarrativeWorkflow._config = config or {}
        logger.info("Initializing NarrativeWorkflow with config: %s", self._config)
        
    @classmethod
    async def initialize_llm(cls) -> LLM:
        """Initialize LLM based on provider and model configuration."""
        if cls._llm is None:
            provider = cls._config.get("provider", "ollama")
            model = cls._config.get("model", "aya-expanse:8b-q6_K")
            
            logger.info(f"Initializing LLM with provider: {provider}, model: {model}")
            
            try:
                if provider == "ollama":
                    cls._llm = Ollama(model=model, temperature=0.8)
                elif provider == "openai":
                    cls._llm = OpenAI(model=model, temperature=0.8)
                elif provider == "anthropic":
                    cls._llm = Anthropic(model=model, temperature=0.8)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
                
                logger.info("LLM initialization successful")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        
        return cls._llm
    
    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def envision_story(
        self, ctx: Context, ev: StartEvent
    ) -> Union[PlanningEvent, StopEvent]:
        """
        Sets up story elements that could naturally emerge, with enhanced plot planning
        """
        logger.info("Starting story envisioning process")
        
        plot = ev.get("plot")
        current_scene = ev.get("current_scene")
        user_action = ev.get("user_action")
        scene_history = ev.get("scene_history", [])
        
        if not all([plot, current_scene]):
            logger.warning("Missing required story elements: plot=%s, current_scene=%s", 
                         bool(plot), bool(current_scene))
            return StopEvent(result="Missing required story elements.")
            
        story_context = StoryContext(plot, current_scene, scene_history)
        logger.info("Created story context with %d historical scenes", len(scene_history))
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = self._resource_manager.get_text("workflow.planner_prompt").format(
                plot=plot,
                history_context=history_context,
                current_scene=current_scene
            )
            
            logger.info("Generating narrative vision")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated narrative vision")
            
            await ctx.set("user_action", user_action)
            
            return PlanningEvent(
                context=story_context,
                narrative_vision=response.text
            )
        except Exception as e:
            logger.error("Error in envision_story: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def generate_response(
        self, ctx: Context, ev: UserInputEvent
    ) -> NarrativeResponseEvent:
        """
        Creates narrative response that flows naturally from the action
        """
        logger.info("Starting response generation for user action")
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(ev.context.scene_history)]) if ev.context.scene_history else "No previous scenes"
            
            prompt = self._resource_manager.get_text("workflow.adapter_prompt").format(
                history_context=history_context,
                current_scene=ev.context.current_scene,
                narrative_vision=ev.narrative_vision,
                user_action=ev.user_action
            )
            
            logger.info("Generating narrative response")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated narrative response")
            
            return NarrativeResponseEvent(
                narrative=response.text,
                original_vision=ev.narrative_vision
            )
        except Exception as e:
            logger.error("Error in generate_response: %s", str(e))
            raise

    @step
    async def process_input(
        self, ctx: Context, ev: PlanningEvent
    ) -> Union[UserInputEvent, StopEvent]:
        """
        Processes user input in context of the story
        """
        logger.info("Processing user input")
        
        user_action = await ctx.get("user_action")
        if not user_action:
            logger.warning("Missing user action in process_input")
            return StopEvent(result="Missing user action.")
            
        return UserInputEvent(
            context=ev.context,
            user_action=user_action,
            narrative_vision=ev.narrative_vision
        )

    @step
    async def format_response(
        self, ctx: Context, ev: NarrativeResponseEvent
    ) -> StopEvent:
        """
        Returns the story response with its original vision
        """
        logger.info("Formatting final response")
        return StopEvent(result={
            "original_vision": ev.original_vision,
            "narrative": ev.narrative
        })

async def generate_narrative(
    plot: str,
    current_scene: str,
    user_action: str,
    scene_history: List[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    logger.info("Starting narrative generation")
    logger.info("Scene history length: %d", len(scene_history) if scene_history else 0)
    
    try:
        workflow = NarrativeWorkflow(config=config or {}, timeout=120)
        
        result = await workflow.run(
            plot=plot,
            current_scene=current_scene,
            user_action=user_action,
            scene_history=scene_history or []
        )
        
        logger.info("Successfully completed narrative generation")
        return result
    except Exception as e:
        logger.error("Error in generate_narrative: %s", str(e))
        raise
