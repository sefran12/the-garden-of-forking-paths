import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

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
    llm: OpenAI = None

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or {}
        logger.info("Initializing NarrativeWorkflow with config: %s", self.config)
        
    @classmethod
    async def initialize_llm(cls):
        if cls.llm is None:
            logger.info("Initializing LLM with model: vanilj/gemma-2-ataraxy-9b:Q6_K")
            try:
                cls.llm = Ollama(model="vanilj/gemma-2-ataraxy-9b:Q6_K", temperature=0.8)
                logger.info("LLM initialization successful")
            except Exception as e:
                logger.error("Failed to initialize LLM: %s", str(e))
                raise
        return cls.llm
    
    @step
    async def envision_story(
        self, ctx: Context, ev: StartEvent
    ) -> PlanningEvent | StopEvent:
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
            
            prompt = f"""
            Given this story:
            World: {plot}

            Previous scenes in chronological order:
            {history_context}

            Current Scene: {current_scene}

            Analyze the story elements and potential developments:

            1. Immediate Scene Elements:
            - [2-3 key physical elements present]
            - [1-2 environmental details that could become significant]
            - [1-2 character states or tensions]

            2. Active Plot Threads:
            - [2-3 ongoing situations or conflicts from previous scenes]
            - [Their current state of development]
            - [Potential ways they could evolve]

            3. Future Developments (Short-term):
            - [2-3 immediate possibilities based on current scene]
            - [Potential consequences of these developments]

            4. Future Developments (Long-term):
            - [2-3 potential major plot developments]
            - [How current elements could lead to these developments]

            Keep elements concrete and connected to established story elements. 
            Focus on building a coherent narrative that flows from past events 
            while setting up future possibilities.
            """
            
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

    @step
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
            
            prompt = f"""
            Story history in chronological order:
            {history_context}

            Current scene:
            {ev.context.current_scene}

            Story elements and potential developments:
            {ev.narrative_vision}

            The character's action:
            {ev.user_action}

            Write what happens next. Focus on:
    1. Clear, concrete sensory details (what is seen, heard, felt)
    2. Specific physical actions and reactions
    3. Direct consequences of the character's choices
    4. Observable changes in the environment or other characters
    5. A specific detail or event that suggests what might happen next

    Important guidelines:
    - Stay grounded in physical reality - avoid metaphors and philosophical musings
    - End on concrete action or observation, not interpretation
    - Show events through direct observation, not narrative commentary
    - Avoid explaining the meaning or significance of events
    - Keep descriptions precise and literal, not flowery or metaphorical
    - No meta-commentary about the nature of reality or deeper meanings

    Example of good ending:
    "The metal door closed behind them with a hollow clang. Down the dim corridor, a faint blue light flickered."

    Example of bad ending:
    "The door seemed to swallow them into its depths, as if the very fabric of reality was shifting. Perhaps the corridor held more than just darkness - it might contain the very essence of their journey."

    Write a clear, direct scene focusing on what actually happens.
            """
            
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
    ) -> UserInputEvent | StopEvent:
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
