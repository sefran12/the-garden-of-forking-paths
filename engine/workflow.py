from dataclasses import dataclass
from typing import Dict, Any, Optional
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

class StoryContext:
    def __init__(self, plot: str, current_scene: str):
        self.plot = plot
        self.current_scene = current_scene
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
        
    @classmethod
    async def initialize_llm(cls):
        if cls.llm is None:
            cls.llm = Ollama(model="vanilj/gemma-2-ataraxy-9b:Q6_K", temperature=0.8)
        return cls.llm
    
    @step
    async def envision_story(
        self, ctx: Context, ev: StartEvent
    ) -> PlanningEvent | StopEvent:
        """
        Sets up story elements that could naturally emerge
        """
        plot = ev.get("plot")
        current_scene = ev.get("current_scene")
        user_action = ev.get("user_action")
        
        if not all([plot, current_scene]):
            return StopEvent(result="Missing required story elements.")
            
        story_context = StoryContext(plot, current_scene)
        llm = await self.initialize_llm()
        
        prompt = f"""
        Given this scene:
        World: {plot}
        Current Scene: {current_scene}

        List potential story elements that could naturally emerge:

        1. What's immediately visible or present?
        - [2-3 key physical elements]
        - [1-2 environmental details]

        2. What might be discovered or noticed?
        - [2-3 possible discoveries]

        3. What could naturally happen next?
        - [2 possible developments]

        Keep each element simple and concrete. Focus on what's immediately 
        relevant rather than long-term plot.
        """
        
        response = await llm.acomplete(prompt)
        await ctx.set("user_action", user_action)
        
        return PlanningEvent(
            context=story_context,
            narrative_vision=response.text
        )

    @step
    async def generate_response(
        self, ctx: Context, ev: UserInputEvent
    ) -> NarrativeResponseEvent:
        """
        Creates narrative response that flows naturally from the action
        """
        llm = await self.initialize_llm()
        
        prompt = f"""
        The scene:
        {ev.context.current_scene}

        Potential elements:
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
        
        response = await llm.acomplete(prompt)
        
        return NarrativeResponseEvent(
            narrative=response.text,
            original_vision=ev.narrative_vision
        )


    @step
    async def process_input(
        self, ctx: Context, ev: PlanningEvent
    ) -> UserInputEvent | StopEvent:
        """
        Processes user input in context of the story
        """
        user_action = await ctx.get("user_action")
        if not user_action:
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
        return StopEvent(result={
            "original_vision": ev.original_vision,
            "narrative": ev.narrative
        })

async def generate_narrative(
    plot: str,
    current_scene: str,
    user_action: str,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    workflow = NarrativeWorkflow(config=config or {}, timeout=120)
    
    result = await workflow.run(
        plot=plot,
        current_scene=current_scene,
        user_action=user_action
    )
    
    return result
