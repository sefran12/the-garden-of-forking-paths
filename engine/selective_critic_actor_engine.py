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
logger = logging.getLogger('selective_critic_actor')

class StoryContext:
    def __init__(self, plot: str, current_scene: str, scene_history: List[str] = None):
        self.plot = plot
        self.current_scene = current_scene
        self.scene_history = scene_history or []

class CriticAnalysisEvent(Event):
    context: StoryContext
    analysis: str
    actor_type: str

class ActorPolicyEvent(Event):
    context: StoryContext
    policy: str
    analysis: str
    actor_type: str

class NarrativeResponseEvent(Event):
    narrative: str
    analysis: str
    policy: str
    actor_type: str

class SelectiveCriticActorWorkflow(Workflow):
    _llm: Optional[LLM] = None
    _config: ClassVar[Dict[str, Any]] = {}

    def __init__(self, *args, config: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        SelectiveCriticActorWorkflow._config = config or {}
        
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
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {str(e)}")
                raise
        
        return cls._llm

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def selective_critic(
        self, ctx: Context, ev: StartEvent
    ) -> Union[CriticAnalysisEvent, StopEvent]:
        """
        Critic analyzes story state and selects appropriate actor type
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
            Analyze this story moment and determine the most appropriate narrative approach:

            World Context:
            {plot}

            Story History:
            {history_context}

            Current Scene:
            {current_scene}

            User Action:
            {user_action}

            First, analyze the current state across these dimensions:

            1. Physical State:
            - Current tangible elements and their states
            - Observable changes from the action
            - Environmental details of significance

            2. Story Momentum:
            - Immediate consequences in play
            - Active story threads and their states
            - Time-sensitive elements or pressures

            3. Narrative Tension:
            - Current conflicts or pressures
            - Contrast between expectations and reality
            - Stakes and risks in play

            Then, determine which actor type would best handle this situation:

            EXPLORATION: For scenes focused on discovery, investigation, or understanding
            CONFLICT: For scenes of direct confrontation or challenge
            INTERACTION: For scenes centered on character relationships or dialogue
            TRANSITION: For scenes bridging major story developments
            REVELATION: For scenes unveiling important information or changes

            Format your response exactly like this:
            ANALYSIS:
            [Your dimensional analysis here]

            SELECTED ACTOR: [EXPLORATION|CONFLICT|INTERACTION|TRANSITION|REVELATION]
            REASON: [Brief explanation of why this actor type is most appropriate]

            Keep your analysis focused on concrete, observable elements.
            Choose the actor type based on the most prominent story needs.
            """
            
            response = await llm.acomplete(prompt)
            
            # Parse the response to extract actor type
            response_text = response.text
            actor_type = "EXPLORATION"  # Default
            
            if "SELECTED ACTOR:" in response_text:
                actor_line = response_text.split("SELECTED ACTOR:")[1].split("\n")[0].strip()
                for possible_type in ["EXPLORATION", "CONFLICT", "INTERACTION", "TRANSITION", "REVELATION"]:
                    if possible_type in actor_line:
                        actor_type = possible_type
                        break
            
            await ctx.set("user_action", user_action)
            
            return CriticAnalysisEvent(
                context=story_context,
                analysis=response_text,
                actor_type=actor_type
            )
        except Exception as e:
            logger.error(f"Error in selective_critic: {str(e)}")
            return StopEvent(result=f"Error in selective_critic: {str(e)}")

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def specialized_actor(
        self, ctx: Context, ev: CriticAnalysisEvent
    ) -> ActorPolicyEvent:
        """
        Selected actor type generates appropriate policy
        """
        try:
            llm = await self.initialize_llm()
            
            history_context = "\n".join([f"Scene {i+1}: {scene}" for i, scene in enumerate(ev.context.scene_history)]) if ev.context.scene_history else "No previous scenes"
            
            # Actor-specific prompts focusing on concrete elements
            actor_prompts = {
                "EXPLORATION": """
                    Focus on physical discovery and environmental detail:
                    - Concrete sensory information about the space
                    - Observable clues and significant details
                    - Physical navigation and investigation methods
                    - Environmental changes and reactions
                    """,
                "CONFLICT": """
                    Focus on tangible tension and physical stakes:
                    - Clear positions and tactical elements
                    - Observable advantages and vulnerabilities
                    - Physical consequences and risks
                    - Environmental factors affecting the conflict
                    """,
                "INTERACTION": """
                    Focus on observable character dynamics:
                    - Physical positioning and body language
                    - Concrete actions and reactions
                    - Environmental influence on interaction
                    - Tangible shifts in relationships
                    """,
                "TRANSITION": """
                    Focus on physical movement and change:
                    - Clear progression of space and time
                    - Observable transformations
                    - Physical connections between states
                    - Environmental shifts and adjustments
                    """,
                "REVELATION": """
                    Focus on tangible discovery and impact:
                    - Physical manifestation of revelations
                    - Observable reactions and consequences
                    - Concrete changes in understanding
                    - Environmental reflection of discovery
                    """
            }
            
            actor_prompt = actor_prompts.get(ev.actor_type, actor_prompts["EXPLORATION"])
            
            prompt = f"""
            As the {ev.actor_type} Actor, create a policy for this story moment:

            World Context:
            {ev.context.plot}

            Story History:
            {history_context}

            Current Scene:
            {ev.context.current_scene}

            Critical Analysis:
            {ev.analysis}

            {actor_prompt}

            Generate a policy that:
            1. Focuses on concrete, physical elements
            2. Sets up clear possibilities without assuming outcomes
            3. Creates natural tension through tangible details
            4. Maintains story momentum through observable elements

            Format your response with these sections:

            ACTIVE ELEMENTS:
            [List key physical elements and their states]

            IMMEDIATE POSSIBILITIES:
            [List concrete developments that could occur]

            SETUP ELEMENTS:
            [List tangible details that suggest future developments]
            """
            
            response = await llm.acomplete(prompt)
            
            return ActorPolicyEvent(
                context=ev.context,
                policy=response.text,
                analysis=ev.analysis,
                actor_type=ev.actor_type
            )
        except Exception as e:
            logger.error("Error in specialized_actor: %s", str(e))
            raise

    @step(retry_policy=retry_policy.ConstantDelayRetryPolicy(maximum_attempts=3, delay=1))
    async def generate_response(
        self, ctx: Context, ev: ActorPolicyEvent
    ) -> NarrativeResponseEvent:
        """
        Generates final narrative response combining policy and action
        """
        try:
            llm = await self.initialize_llm()
            
            user_action = await ctx.get("user_action")
            if not user_action:
                raise ValueError("User action not found in context")
            
            prompt = f"""
            Generate the next scene based on:

            Current Scene:
            {ev.context.current_scene}

            Actor Policy ({ev.actor_type}):
            {ev.policy}

            User Action:
            {user_action}

            Create a scene that:
            1. Shows clear physical consequences of the action
            2. Uses concrete sensory details
            3. Creates natural tension through tangible elements
            4. Sets up future possibilities through observable details

            Guidelines:
            - Focus on what can be directly observed
            - Show developments through physical details
            - Use precise, literal descriptions
            - End on concrete details that suggest possibilities
            - Avoid explaining meaning or significance
            """
            
            response = await llm.acomplete(prompt)
            
            return NarrativeResponseEvent(
                narrative=response.text,
                analysis=ev.analysis,
                policy=ev.policy,
                actor_type=ev.actor_type
            )
        except Exception as e:
            logger.error("Error in generate_response: %s", str(e))
            raise

    @step
    async def format_response(
        self, ctx: Context, ev: NarrativeResponseEvent
    ) -> StopEvent:
        """
        Returns the final formatted response
        """
        original_vision = f"""
CRITIC ANALYSIS:
{ev.analysis}

{ev.actor_type} ACTOR POLICY:
{ev.policy}
"""
        
        return StopEvent(result={
            "original_vision": original_vision,
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
        workflow = SelectiveCriticActorWorkflow(config=config or {}, timeout=120)
        
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
