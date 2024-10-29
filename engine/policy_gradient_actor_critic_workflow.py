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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('actor_critic_workflow')

class StoryContext:
    def __init__(self, plot: str, current_scene: str, scene_history: List[str] = None):
        self.plot = plot
        self.current_scene = current_scene
        self.scene_history = scene_history or []
        self.narrative_threads = {}

class PolicyEvent(Event):
    context: StoryContext
    policy: str

class UserActionEvent(Event):
    context: StoryContext
    user_action: str
    policy: str

class CriticResponseEvent(Event):
    narrative: str
    original_policy: str

class PolicyGradientActorCriticWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        ActorCriticWorkflow._config = config or {}
        logger.info("Initializing ActorCriticWorkflow with config: %s", self._config)
        
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
    async def actor_step(
        self, ctx: Context, ev: StartEvent
    ) -> Union[PolicyEvent, StopEvent]:
        """
        Actor: Analyzes narrative dimensions and suggests shifts
        """
        logger.info("Starting actor analysis")
        
        plot = ev.get("plot")
        current_scene = ev.get("current_scene")
        user_action = ev.get("user_action")
        scene_history = ev.get("scene_history", [])
        
        if not all([plot, current_scene]):
            logger.warning("Missing required story elements")
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

            As the Actor, analyze the narrative dimensions and their needed movements.
            Focus on concrete, sensory details and clear story elements.
            Keep your analysis grounded in physical reality and tangible developments.

            POLICY:
            1. Material Dimension:
            - List key physical elements in motion
            - Note significant positions and states
            - Identify environmental forces at play
            - Mark objects or tools of significance
            Suggest how these material elements need to shift, evolve, or transform

            2. Tension Dimension:
            - Highlight active conflicts and pressures
            - List ongoing situations requiring movement
            - Note time-sensitive elements
            - Identify stakes and risks in flux
            Describe how these tensions should escalate, resolve, or transform

            3. Possibility Dimension:
            - List potential story paths now available
            - Note resources or advantages in play
            - Identify promising connections between elements
            - Mark emerging opportunities or threats
            Suggest which possibilities should open, close, or transform

            For each dimension:
            - Ground your analysis in specific details
            - Focus on elements that drive concrete action
            - Explain the desired direction of change
            - Note why this movement serves the story
            
            End each dimensional analysis with clear, physical imagery that captures its desired movement.
            """
            
            logger.info("Generating dimensional analysis")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated analysis")
            
            await ctx.set("user_action", user_action)
            
            return PolicyEvent(
                context=story_context,
                policy=response.text
            )
            
        except Exception as e:
            logger.error("Error in actor_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def critic_step(
        self, ctx: Context, ev: UserActionEvent
    ) -> CriticResponseEvent:
        """
        Critic: Evaluates action against dimensional analysis and crafts response
        """
        logger.info("Starting critic evaluation and response")
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(ev.context.scene_history)]) if ev.context.scene_history else "No previous scenes"
            
            prompt = f"""
            Story history in chronological order:
            {history_context}

            Current scene:
            {ev.context.current_scene}

            Actor's dimensional analysis:
            {ev.policy}

            User's action:
            {ev.user_action}

            As the Critic, craft a response that weaves together the narrative dimensions based on the user's action.
            Focus on concrete, sensory details and clear physical imagery.

            Action Analysis:
            - How does this action affect each narrative dimension?
            - Which dimensional movements does it support or resist?
            - What new possibilities emerge from dimensional interactions?
            - How do the dimensions push and pull against each other?
            - What unexpected combinations or tensions arise?
            - Which physical details best capture these movements?

            Response:
            [Craft the next scene with rich sensory detail:]
            - Ground every movement in concrete physical reality
            - Show how dimensions interact through specific details
            - Use precise sensory information
            - Capture character actions and reactions physically
            - Demonstrate environmental changes clearly
            - Thread dimensional shifts through tangible elements
            - End with physical details that suggest future movement

            Keep the scene anchored in observable reality.
            Let dimensional shifts emerge through concrete details.
            End with clear physical elements in motion.
            """
            
            logger.info("Generating critic response")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated response")
            
            # Extract sections
            full_response = response.text
            try:
                analysis_section = full_response.split("Response:")[0].replace("Action Analysis:", "").strip()
                response_section = full_response.split("Response:")[1].strip()
            except IndexError:
                logger.warning("Could not split response, using full text")
                analysis_section = "Analysis parsing failed"
                response_section = full_response
            
            # Combine Actor's policy and Critic's analysis
            combined_vision = f"""ACTOR'S DIMENSIONAL ANALYSIS:
{ev.policy}

CRITIC'S DIMENSIONAL WEAVING:
{analysis_section}"""
            
            return CriticResponseEvent(
                narrative=response_section,
                original_policy=combined_vision
            )
        except Exception as e:
            logger.error("Error in critic_step: %s", str(e))
            raise

    @step
    async def process_action(
        self, ctx: Context, ev: PolicyEvent
    ) -> Union[UserActionEvent, StopEvent]:
        """
        Processes user action in context of the dimensional analysis
        """
        logger.info("Processing user action")
        
        user_action = await ctx.get("user_action")
        if not user_action:
            logger.warning("Missing user action in process_action")
            return StopEvent(result="Missing user action.")
            
        return UserActionEvent(
            context=ev.context,
            user_action=user_action,
            policy=ev.policy
        )

    @step
    async def format_response(
        self, ctx: Context, ev: CriticResponseEvent
    ) -> StopEvent:
        """
        Returns the story response with its original analysis
        """
        logger.info("Formatting final response")
        return StopEvent(result={
            "original_vision": ev.original_policy,
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
        workflow = ActorCriticWorkflow(config=config or {}, timeout=120)
        
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