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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimizing_critic_actor')

class StoryContext:
    def __init__(self, plot: str, current_scene: str, scene_history: List[str] = None):
        self.plot = plot
        self.current_scene = current_scene
        self.scene_history = scene_history or []

class CriticAnalysisEvent(Event):
    context: StoryContext
    analysis: str
    user_action: str

class ActorResponseEvent(Event):
    narrative: str
    analysis: str

class OptimizingCriticActorWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        OptimizingCriticActorWorkflow._config = config or {}
        logger.info("Initializing OptimizingCriticActorWorkflow with config: %s", self._config)

    @classmethod
    async def initialize_llm(cls) -> LLM:
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
    async def critic_step(
        self, ctx: Context, ev: StartEvent
    ) -> Union[CriticAnalysisEvent, StopEvent]:
        """
        Critic: Analyzes story state and trajectory, seeing both scene and action
        """
        logger.info("Starting critic analysis")
        
        plot = ev.get("plot")
        current_scene = ev.get("current_scene")
        user_action = ev.get("user_action")
        scene_history = ev.get("scene_history", [])
        
        if not all([plot, current_scene, user_action]):
            logger.warning("Missing required story elements")
            return StopEvent(result="Missing required story elements.")
            
        story_context = StoryContext(plot, current_scene, scene_history)
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = f"""
            Given this story:
            World: {plot}

            Previous scenes in chronological order:
            {history_context}

            Current Scene: {current_scene}

            User Action: {user_action}

            Analyze the story's trajectory and needs. Format your response with clear sections:

            1. Current Elements:
            - [List 2-3 key physical elements in play]
            - [Note 1-2 significant character states]
            - [Identify 1-2 active environmental details]

            2. Narrative Pacing:
            - [Assess speed of development from immediate to long-term events]
            - [Note any rushed or stagnant story elements]
            - [Identify optimal rhythm for current story phase]

            3. Context Richness:
            - [List established world elements being utilized]
            - [Note unexplored areas with potential]
            - [Identify opportunities for deeper context]

            4. Plot Complexity:
            - [Analyze independence of current plot threads]
            - [Note how predictable current developments are]
            - [Identify potential for unexpected connections]

            5. Development Needs:
            - [List 2-3 elements needing immediate attention]
            - [Note 1-2 longer-term elements to develop]
            - [Identify optimal balance between immediate and future focus]

            Keep analysis concrete and specific.
            Focus on tangible elements that can shape the story's development.
            """
            
            logger.info("Generating critic analysis")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated critic analysis")
            
            return CriticAnalysisEvent(
                context=story_context,
                analysis=response.text,
                user_action=user_action
            )
        except Exception as e:
            logger.error("Error in critic_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def actor_step(
        self, ctx: Context, ev: CriticAnalysisEvent
    ) -> ActorResponseEvent:
        """
        Actor: Creates next scene guided by critic's analysis, without seeing user action
        """
        logger.info("Starting actor generation")
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(ev.context.scene_history)]) if ev.context.scene_history else "No previous scenes"
            
            prompt = f"""
            Given this story context:
            World: {ev.context.plot}

            Previous scenes:
            {history_context}

            Current Scene: {ev.context.current_scene}

            Critic's Analysis:
            {ev.analysis}

            Continue the story directly, focusing on:

            1. Clear, concrete sensory details (what is seen, heard, felt)
            2. Specific physical actions and reactions
            3. Observable changes in the environment or characters
            4. Direct consequences of events
            5. Details that connect to established elements

            Important guidelines:
            - Begin the narrative without any preamble or phrases like "Here's the next scene:"
            - Stay grounded in physical reality
            - Show events through direct observation
            - Keep descriptions precise and literal
            - End on concrete details that suggest future possibilities
            - Maintain established pacing and complexity
            - Develop identified story needs

            Write a clear, direct scene focusing on what actually happens.
            """
            
            logger.info("Generating actor response")
            response = await llm.acomplete(prompt, max_tokens=1024*4)
            logger.info("Successfully generated actor response")
            
            return ActorResponseEvent(
                narrative=response.text,
                analysis=ev.analysis
            )
        except Exception as e:
            logger.error("Error in actor_step: %s", str(e))
            raise


    @step
    async def format_response(
        self, ctx: Context, ev: ActorResponseEvent
    ) -> StopEvent:
        """
        Returns the final formatted response
        """
        logger.info("Formatting final response")
        return StopEvent(result={
            "original_vision": ev.analysis,
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
        workflow = OptimizingCriticActorWorkflow(config=config or {}, timeout=120)
        
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
