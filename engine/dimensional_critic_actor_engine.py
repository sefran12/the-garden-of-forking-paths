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
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms.llm import LLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dimensional_critic_actor')

class StoryContext:
    def __init__(self, plot: str, current_scene: str, scene_history: List[str] = None):
        self.plot = plot
        self.current_scene = current_scene
        self.scene_history = scene_history or []

class CriticAnalysisEvent(Event):
    context: StoryContext
    dimensional_analysis: str
    user_action: str

class ActorResponseEvent(Event):
    narrative: str
    original_analysis: str

class DimensionalCriticActorWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        DimensionalCriticActorWorkflow._config = config or {}
        
    @classmethod
    async def initialize_llm(cls) -> LLM:
        if cls._llm is None:
            provider = cls._config.get("provider", "ollama")
            model = cls._config.get("model", "aya-expanse:8b-q6_K")
            
            try:
                if provider == "ollama":
                    cls._llm = Ollama(model=model, temperature=0.8)
                elif provider == "openai":
                    cls._llm = OpenAI(model=model, temperature=0.8)
                elif provider == "anthropic":
                    cls._llm = Anthropic(model=model, temperature=0.8)
                elif provider == "together":
                    cls._llm = TogetherLLM(model=model, temperature=0.8)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        
        return cls._llm

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def critic_analysis(
        self, ctx: Context, ev: StartEvent
    ) -> Union[CriticAnalysisEvent, StopEvent]:
        """
        Critic analyzes the current state across multiple dimensions
        """
        plot = ev.get("plot")
        current_scene = ev.get("current_scene")
        user_action = ev.get("user_action")
        scene_history = ev.get("scene_history", [])
        
        if not all([plot, current_scene, user_action]):
            return StopEvent(result="Missing required story elements.")
            
        story_context = StoryContext(plot, current_scene, scene_history)
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = f"""
            Story Context:
            World: {plot}
            Previous Scenes: {history_context}
            Current Scene: {current_scene}
            User Action: {user_action}

            Analyze the current story state across these dimensions, focusing on concrete manifestations:

            1. Physical Dimension:
            - Current tangible elements and their states
            - Observable changes from the action
            - Physical constraints and opportunities
            - Environmental details that gained/lost significance

            2. Temporal Dimension:
            - Immediate consequences vs potential long-term effects
            - Pacing (rate of progression from immediate to long-term)
            - Time-sensitive elements or pressures
            - Historical connections to past scenes

            3. Causal Dimension:
            - Direct cause-effect relationships
            - Emerging chain reactions
            - Broken or reinforced patterns
            - New possibilities created/eliminated

            4. Contextual Dimension:
            - Active story elements and their current roles
            - Shifting relationships between elements
            - Environmental influences on action
            - Resource availability and constraints

            Keep analysis focused on concrete, observable elements.
            Highlight contrasts between expected and actual outcomes.
            Note specific physical details that carry story weight.
            """
            
            response = await llm.acomplete(prompt)
            
            return CriticAnalysisEvent(
                context=story_context,
                dimensional_analysis=response.text,
                user_action=user_action
            )
        except Exception as e:
            logger.error("Error in critic_analysis: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def actor_response(
        self, ctx: Context, ev: CriticAnalysisEvent
    ) -> ActorResponseEvent:
        """
        Actor generates response based on critic's analysis without seeing user action
        """
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(ev.context.scene_history)]) if ev.context.scene_history else "No previous scenes"
            
            # Note: We explicitly exclude user_action here to preserve the key source of narrative richness
            prompt = f"""
            Story Context:
            World: {ev.context.plot}
            Previous Scenes: {history_context}
            Current Scene: {ev.context.current_scene}

            Dimensional Analysis:
            {ev.dimensional_analysis}

            Generate the next scene focusing on concrete physical details and tangible developments.
            Structure your response in these sections:

            1. Immediate Physical State:
            - Clear sensory details (what is seen, heard, felt)
            - Specific positions and movements
            - Observable environmental changes
            - Concrete object states and interactions

            2. Active Developments:
            - Direct consequences playing out
            - New physical situations emerging
            - Tangible shifts in relationships
            - Observable changes in story elements

            3. Future Setup:
            - Physical details that suggest possibilities
            - New environmental factors introduced
            - Concrete elements gaining significance
            - Observable tensions or pressures building

            Guidelines:
            - Use precise, literal descriptions
            - Focus on what can be directly observed
            - Show developments through physical details
            - End on concrete details that suggest future possibilities
            - Avoid metaphors or abstract concepts
            - Don't explain meaning or significance
            """
            
            response = await llm.acomplete(prompt)
            
            return ActorResponseEvent(
                narrative=response.text,
                original_analysis=ev.dimensional_analysis
            )
        except Exception as e:
            logger.error("Error in actor_response: %s", str(e))
            raise

    @step
    async def format_response(
        self, ctx: Context, ev: ActorResponseEvent
    ) -> StopEvent:
        """
        Returns the final narrative with original analysis
        """
        return StopEvent(result={
            "original_vision": ev.original_analysis,  # Changed from original_analysis to original_vision
            "narrative": ev.narrative
        })

async def generate_narrative(
    plot: str,
    current_scene: str,
    user_action: str,
    scene_history: List[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    try:
        workflow = DimensionalCriticActorWorkflow(config=config or {}, timeout=120)
        
        result = await workflow.run(
            plot=plot,
            current_scene=current_scene,
            user_action=user_action,
            scene_history=scene_history or []
        )
        
        return result
    except Exception as e:
        logger.error("Error in generate_narrative: %s", str(e))
        raise
