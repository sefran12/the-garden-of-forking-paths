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
logger = logging.getLogger('timescale_actor_critic_workflow')

@dataclass
class StoryContext:
    plot: str
    current_scene: str
    scene_history: List[str] = None
    long_term_policy: Optional[str] = None
    short_term_policy: Optional[str] = None
    merged_policy: Optional[str] = None
    narrative_threads: Dict[str, Any] = None

@dataclass
class LongTermPolicyEvent(Event):
    context: StoryContext
    long_term_policy: str

@dataclass
class ShortTermPolicyEvent(Event):
    context: StoryContext
    short_term_policy: str

@dataclass
class MergedPolicyEvent(Event):
    context: StoryContext
    merged_policy: str

@dataclass
class UserActionEvent(Event):
    context: StoryContext
    user_action: str
    merged_policy: str

@dataclass
class CriticResponseEvent(Event):
    narrative: str
    original_policy: str

class TimescaleActorCriticWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        TimescaleActorCriticWorkflow._config = config or {}
        logger.info("Initializing TimescaleActorCriticWorkflow with config: %s", self._config)
        
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
    async def long_term_actor_step(
        self, ctx: Context, ev: StartEvent
    ) -> Union[LongTermPolicyEvent, StopEvent]:
        """
        Long-Term Actor: Generates a long-term narrative policy based on the overall plot and scene history.
        """
        logger.info("Starting long-term actor policy generation")
        
        plot = ev.get("plot")
        scene_history = ev.get("scene_history", [])
        
        if not plot:
            logger.warning("Missing plot information for long-term policy generation.")
            return StopEvent(result="Missing plot information.")
        
        story_context = StoryContext(plot=plot, current_scene="", scene_history=scene_history)
        logger.info("Created story context for long-term policy with %d historical scenes", len(scene_history))
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = f"""
            Given this story plot:
            World: {plot}

            Previous scenes in chronological order:
            {history_context}

            As the Long-Term Actor, generate a long-term narrative policy that outlines the overarching plot developments and story arcs.
            Focus on high-level themes, character development, and major events that should occur throughout the story.
            Format your response with clear sections:

            POLICY:
            1. Major Plot Points:
            - List key events that drive the story forward
            - Outline the progression of the main conflict
            - Define the resolution path for the story

            2. Character Arcs:
            - Describe the development paths for main characters
            - Highlight significant changes in character motivations and relationships

            3. Thematic Elements:
            - Identify the core themes to be explored
            - Ensure thematic consistency throughout the narrative

            Keep each point specific and actionable to guide the overall story progression.
            """
            
            logger.info("Generating long-term narrative policy")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated long-term narrative policy")
            
            story_context.long_term_policy = response.text
            return LongTermPolicyEvent(
                context=story_context,
                long_term_policy=response.text
            )
        except Exception as e:
            logger.error("Error in long_term_actor_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def short_term_actor_step(
        self, ctx: Context, ev: LongTermPolicyEvent
    ) -> Union[ShortTermPolicyEvent, StopEvent]:
        """
        Short-Term Actor: Generates a short-term narrative policy based on the current scene and long-term policy.
        """
        logger.info("Starting short-term actor policy generation")
        
        plot = ev.context.plot
        current_scene = ev.context.current_scene
        scene_history = ev.context.scene_history
        long_term_policy = ev.long_term_policy
        
        if not current_scene:
            logger.warning("Missing current scene for short-term policy generation.")
            return StopEvent(result="Missing current scene.")
        
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = f"""
            Given this story plot:
            World: {plot}

            Previous scenes in chronological order:
            {history_context}

            Current Scene: {current_scene}

            Long-Term Policy:
            {long_term_policy}

            As the Short-Term Actor, generate a short-term narrative policy that outlines immediate story developments.
            Focus on concrete actions, character interactions, and events that advance the current scene.
            Ensure that these short-term actions contribute to the long-term narrative goals.
            Format your response with clear sections:

            POLICY:
            1. Immediate Actions:
            - Describe actions characters will take
            - Define interactions between characters
            - Outline environmental changes or events

            2. Scene Objectives:
            - Specify the goals for the current scene
            - Ensure objectives align with long-term plot points

            3. Transitional Elements:
            - Identify elements that bridge the current scene to future developments
            - Ensure smooth narrative progression

            Keep each point specific and actionable to guide the immediate story progression.
            """
            
            logger.info("Generating short-term narrative policy")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated short-term narrative policy")
            
            ev.context.short_term_policy = response.text
            return ShortTermPolicyEvent(
                context=ev.context,
                short_term_policy=response.text
            )
        except Exception as e:
            logger.error("Error in short_term_actor_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def pacing_step(
        self, ctx: Context, ev: ShortTermPolicyEvent
    ) -> Union[MergedPolicyEvent, StopEvent]:
        """
        Pacing Mechanism: Merges short-term and long-term policies to create a paced policy.
        """
        logger.info("Starting pacing to merge short-term and long-term policies")
        
        long_term_policy = ev.context.long_term_policy
        short_term_policy = ev.short_term_policy
        
        if not long_term_policy or not short_term_policy:
            logger.warning("Missing policies for pacing.")
            return StopEvent(result="Missing policies for pacing.")
        
        try:
            llm = await self.initialize_llm()
            
            prompt = f"""
            Long-Term Policy:
            {long_term_policy}

            Short-Term Policy:
            {short_term_policy}

            As the Pacing Mechanism, merge the short-term policy into the long-term policy to create a paced policy.
            Ensure that immediate actions contribute to long-term goals and maintain narrative coherence.
            Format your response as a single merged policy.

            MERGED POLICY:
            """
            
            logger.info("Merging policies for pacing")
            response = await llm.acomplete(prompt)
            logger.info("Successfully merged policies")
            
            ev.context.merged_policy = response.text
            return MergedPolicyEvent(
                context=ev.context,
                merged_policy=response.text
            )
        except Exception as e:
            logger.error("Error in pacing_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def process_user_action_step(
        self, ctx: Context, ev: MergedPolicyEvent
    ) -> Union[UserActionEvent, StopEvent]:
        """
        Processes user action in the context of the merged policy.
        """
        logger.info("Processing user action with merged policy")
        
        user_action = ctx.get("user_action")
        if not user_action:
            logger.warning("Missing user action in process_user_action_step")
            return StopEvent(result="Missing user action.")
        
        try:
            # Optionally, adjust the merged policy based on user action here
            # For simplicity, we pass it directly
            return UserActionEvent(
                context=ev.context,
                user_action=user_action,
                merged_policy=ev.merged_policy
            )
        except Exception as e:
            logger.error("Error in process_user_action_step: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def critic_step(
        self, ctx: Context, ev: UserActionEvent
    ) -> CriticResponseEvent:
        """
        Critic: Evaluates the merged policy and user action, then generates the next narrative scene.
        """
        logger.info("Starting critic evaluation and response generation")
        
        try:
            llm = await self.initialize_llm()
            
            plot = ev.context.plot
            scene_history = ev.context.scene_history
            user_action = ev.user_action
            merged_policy = ev.merged_policy
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(scene_history)]) if scene_history else "No previous scenes"
            
            prompt = f"""
            Story plot:
            World: {plot}

            Story history in chronological order:
            {history_context}

            User's Action:
            {user_action}

            Merged Policy:
            {merged_policy}

            As the Critic, evaluate the impact of the user's action on the current state.
            Ensure that the action aligns with both short-term and long-term policies.
            Provide an analysis and generate the next scene based on this evaluation.
            Format your response in two clear sections with the given names "Action Analysis" and "Response":

            Action Analysis:
            - How does this action affect the current elements?
            - What immediate changes does it cause?
            - What new possibilities does it create?
            - Which existing tensions shift or develop?
            - What new practical situations emerge?
            - What potential consequences emerge?

            Response:
            [Write the next scene with clear physical detail:]
            - Specific sensory information
            - Concrete actions and reactions
            - Observable environmental changes
            - Direct consequences of choices
            - Clear character movements and states
            - Precise details that connect to established elements
            - End on concrete details that suggest future possibilities

            Keep the scene grounded in physical reality.
            End with a clear, observable development.
            """
            
            logger.info("Generating critic response")
            response = await llm.acomplete(prompt)
            logger.info("Successfully generated critic response")
            
            # Extract sections
            full_response = response.text
            try:
                analysis_section = full_response.split("Response:")[0].replace("Action Analysis:", "").strip()
                response_section = full_response.split("Response:")[1].strip()
            except IndexError:
                logger.warning("Could not split response, using full text")
                analysis_section = "Analysis parsing failed"
                response_section = full_response
            
            # Combine merged policy and Critic's analysis for original_vision
            combined_vision = f"""MERGED POLICY:
{merged_policy}

CRITIC ANALYSIS:
{analysis_section}"""
            
            return CriticResponseEvent(
                narrative=response_section,
                original_policy=combined_vision
            )
        except Exception as e:
            logger.error("Error in critic_step: %s", str(e))
            raise

    @step
    async def format_response_step(
        self, ctx: Context, ev: CriticResponseEvent
    ) -> StopEvent:
        """
        Returns the story response with its original policy.
        """
        logger.info("Formatting final response")
        return StopEvent(result={
            "original_vision": ev.original_policy,  # Contains merged policy and critic analysis
            "narrative": ev.narrative
        })

    @step
    async def update_story_context_step(
        self, ctx: Context, ev: CriticResponseEvent
    ) -> Union[StartEvent, StopEvent]:
        """
        Updates the story context with the new scene and prepares for the next iteration.
        """
        logger.info("Updating story context with the new scene")
        
        narrative = ev.narrative
        scene_history = ctx.get("scene_history", [])
        scene_history.append(narrative)
        
        # Update the current scene to the new narrative
        ctx.set("current_scene", narrative)
        ctx.set("scene_history", scene_history)
        
        # Check termination condition (for example, a maximum number of scenes)
        max_scenes = self._config.get("max_scenes", 10)
        if len(scene_history) >= max_scenes:
            logger.info("Reached maximum number of scenes: %d", max_scenes)
            return StopEvent(result="Story complete.")
        
        # Continue the workflow
        return StartEvent()

    async def run_workflow(self, plot: str, current_scene: str, user_action: str, scene_history: List[str] = None) -> Dict[str, Any]:
        logger.info("Starting timescale-aware narrative generation")
        logger.info("Initial scene history length: %d", len(scene_history) if scene_history else 0)
        
        try:
            # Initialize the workflow with the starting event
            start_event = StartEvent(
                plot=plot,
                current_scene=current_scene,
                user_action=user_action,
                scene_history=scene_history or []
            )
            result = await self.run(start_event)
            logger.info("Successfully completed narrative generation")
            return result
        except Exception as e:
            logger.error("Error in run_workflow: %s", str(e))
            raise

async def generate_timescale_narrative(
    plot: str,
    current_scene: str,
    user_action: str,
    scene_history: List[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    logger.info("Starting timescale-aware narrative generation")
    logger.info("Scene history length: %d", len(scene_history) if scene_history else 0)
    
    try:
        workflow = TimescaleActorCriticWorkflow(config=config or {}, timeout=300)
        
        result = await workflow.run_workflow(
            plot=plot,
            current_scene=current_scene,
            user_action=user_action,
            scene_history=scene_history or []
        )
        
        logger.info("Successfully completed timescale-aware narrative generation")
        return result
    except Exception as e:
        logger.error("Error in generate_timescale_narrative: %s", str(e))
        raise
