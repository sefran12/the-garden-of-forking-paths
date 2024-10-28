import os
import logging
import traceback
from shiny import App, ui, reactive, render
from app_utils import load_dotenv
from adapter.adapter import WorkflowAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('narrative_app')

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Available models configuration
AVAILABLE_MODELS = [
    # Ollama Models
    {"name": "mistral-nemo:12b", "provider": "ollama", "size": "7.1 GB"},
    {"name": "aya-expanse:8b-q6_K", "provider": "ollama", "size": "6.6 GB"},
    {"name": "technobyte/arliai-rpmax-12b-v1.1:q4_k_m", "provider": "ollama", "size": "7.5 GB"},
    {"name": "michaelbui/nemomix-unleashed-12b:q4-k-m", "provider": "ollama", "size": "7.5 GB"},
    {"name": "jean-luc/tiger-gemma-9b-v3:q6_K", "provider": "ollama", "size": "7.6 GB"},
    {"name": "deepseek-coder-v2:16b-lite-base-q5_K_M", "provider": "ollama", "size": "11 GB"},
    {"name": "qwen2.5:14b-instruct-q5_K_M", "provider": "ollama", "size": "10 GB"},
    {"name": "mistral-small:latest", "provider": "ollama", "size": "12 GB"},
    {"name": "bespoke-minicheck:latest", "provider": "ollama", "size": "4.7 GB"},
    {"name": "minicpm-v:8b-2.6-q8_0", "provider": "ollama", "size": "9.1 GB"},
    {"name": "vanilj/gemma-2-ataraxy-9b:Q6_K", "provider": "ollama", "size": "7.6 GB"},
    {"name": "llama3.2:3b-instruct-q8_0", "provider": "ollama", "size": "3.4 GB"},
    {"name": "qwen2.5-coder:7b-instruct-q8_0", "provider": "ollama", "size": "8.1 GB"},
    {"name": "qwen2-math:7b-instruct-q8_0", "provider": "ollama", "size": "8.1 GB"},
    
    # OpenAI Models
    {"name": "gpt-4o", "provider": "openai", "size": "N/A"},
    {"name": "gpt-4o-mini", "provider": "openai", "size": "N/A"},
    
    # Anthropic Models
    {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic", "size": "N/A"},
    {"name": "claude-3-opus-20240229", "provider": "anthropic", "size": "N/A"},
    {"name": "claude-3-sonnet-20240229", "provider": "anthropic", "size": "N/A"},
    {"name": "claude-3-haiku-20240307", "provider": "anthropic", "size": "N/A"}
]

# Available workflow types
WORKFLOW_TYPES = [
    ("plan-adapt", "Plan & Adapt - Classic planning with adaptation"),
    ("actor-critic", "Actor-Critic - Policy-based narrative generation")
]

# Group models by provider for UI
MODELS_BY_PROVIDER = {}
for model in AVAILABLE_MODELS:
    provider = model["provider"]
    if provider not in MODELS_BY_PROVIDER:
        MODELS_BY_PROVIDER[provider] = []
    display_name = f"{model['name']} ({model['size']})" if model['size'] != "N/A" else model['name']
    MODELS_BY_PROVIDER[provider].append((model['name'], display_name))

class ChatController:
    def __init__(self, input, chat, adapter_rv):
        self.input = input
        self.chat = chat
        self.adapter_rv = adapter_rv
        
    def get_model_info(self):
        return {
            "provider": self.input.model_provider(),
            "model": self.input.model_select(),
            "workflow_type": self.input.workflow_type()
        }
            
    async def new_game(self):
        try:
            adapter = self.adapter_rv.get()
            await self.chat.clear_messages()
            
            welcome_message = {
                "content": """Welcome to the Interactive Narrative! 
                Type your character's actions to see how the story unfolds.
                Each response will become your new current scene, building the narrative.""",
                "role": "assistant"
            }
            initial_scene = {
                "content": self.input.current_scene(),
                "role": "assistant"
            }
            
            await self.chat.append_message(welcome_message)
            await self.chat.append_message(initial_scene)
            
            adapter.create_initial_state(
                plot=self.input.plot(),
                current_scene=self.input.current_scene(),
                chat_messages=[welcome_message, initial_scene],
                scene_history=[]
            )
            
            ui.notification_show("New game started successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to start new game: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def update_game(self):
        try:
            adapter = self.adapter_rv.get()
            if not adapter.current_state:
                ui.notification_show("No active game to update. Please start a new game first.", type="warning")
                return
                
            adapter.current_state.plot = self.input.plot()
            adapter.current_state.current_scene = self.input.current_scene()
            
            ui.notification_show("Game settings updated successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to update game: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def save_state(self):
        try:
            adapter = self.adapter_rv.get()
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in self.chat.messages()
            ]
            adapter.current_state.chat_messages = messages
            
            with ui.Progress(min=0, max=3) as p:
                p.set(value=0, message="Generating story metadata...", 
                      detail="Creating story name and summaries...")
                
                config = self.get_model_info()
                save_path = await adapter.save_state(workflow_config=config)
                
                p.set(value=2, message="Finalizing save...", 
                      detail="Updating save list...")
                
                update_save_list(adapter)
                
                p.set(value=3, message="Save complete!")
            
            ui.notification_show(f"State saved successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to save state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def load_state(self, scenes_rv, rv):
        try:
            adapter = self.adapter_rv.get()
            selected_save = self.input.save_select()
            if not selected_save:
                return
                
            state = adapter.load_state(os.path.join("saves", selected_save))
            
            ui.update_text_area("plot", value=state.plot)
            ui.update_text_area("current_scene", value=state.current_scene)
            scenes_rv.set(state.scene_history)
            
            # When loading a game, just load the saved messages without adding an initial scene
            await self.chat.clear_messages()
            for msg in state.chat_messages:
                await self.chat.append_message(msg)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
                
            ui.notification_show("State loaded successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to load state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def regenerate_scene(self, scenes_rv, rv):
        try:
            adapter = self.adapter_rv.get()
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in self.chat.messages()[:-1]
            ]
            
            config = self.get_model_info()
            state = await adapter.regenerate_current_state(
                chat_messages=messages,
                max_scenes=int(self.input.max_history()),
                workflow_config=config
            )
            
            ui.update_text_area("current_scene", value=state.current_scene)
            scenes_rv.set(state.scene_history)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
                
            await self.chat.clear_messages()
            for msg in messages:
                await self.chat.append_message(msg)
            
            await self.chat.append_message({
                "content": state.current_scene,
                "role": "assistant"
            })
            
            ui.notification_show("Scene regenerated successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to regenerate scene: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.chat.append_message({
                "content": f"Error: {error_msg}",
                "role": "assistant"
            })
            
    async def handle_user_action(self, scenes_rv, rv):
        try:
            adapter = self.adapter_rv.get()
            user_action = self.chat.user_input()
            logger.info("Received user action: %s", user_action)
            
            current_history = scenes_rv.get()
            current_history.append(self.input.current_scene())
            scenes_rv.set(current_history)
            logger.info("Updated scene history, current length: %d", len(current_history))
            
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in self.chat.messages()
            ]
            
            config = self.get_model_info()
            
            state = await adapter.generate_next_state(
                user_action=user_action,
                chat_messages=messages,
                max_scenes=int(self.input.max_history()),
                workflow_config=config
            )
            
            ui.update_text_area("current_scene", value=state.current_scene)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
            
            await self.chat.append_message({
                "content": state.current_scene,
                "role": "assistant"
            })
                
        except Exception as e:
            error_msg = f"Error in chat handler: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.chat.append_message({
                "content": f"Error: {error_msg}",
                "role": "assistant"
            })

# Define the UI
app_ui = ui.page_fillable(
    ui.panel_title("Interactive Narrative Chat"),
    ui.navset_tab(
        ui.nav_panel(
            "Story Settings",
            ui.input_select(
                "workflow_type",
                "Select Narrative Engine:",
                choices=dict(WORKFLOW_TYPES)
            ),
            ui.input_select(
                "model_provider",
                "Select Provider:",
                choices=list(MODELS_BY_PROVIDER.keys())
            ),
            ui.input_select(
                "model_select",
                "Select Language Model:",
                choices=[]  # Will be populated based on provider selection
            ),
            ui.input_text_area(
                id="plot",
                label="Story World/Plot:",
                height="100px",
                value="""The city-state of Kal-shalà stands in the shadow of the ancient Nameless Emperor's throne. Here, magic and technology merge – cybernetic skyscrapers rise beside the Emperor's Palace-Temple, all turning with the Gearwheel of Fate.
The streets hold both old and new. The Ninnuei artifacts and Rusty Cauldron relics share space with AI programs and augmented citizens. In the hyper-district, holographic ads light the night, while beneath the streets, the Spirit Network carries encrypted data through the city's foundations.
The ruling Dynasty, descended from the Emperor's Court, governs through a mix of ancient rites and modern methods. Among the citizens walk street urchins who shape the city's destiny, cyborg oracles reading futures in holograms, technomancers coding spells, and scholars piecing together forgotten histories.
Children play with digital ghosts of ancient beasts while rogue AIs take the forms of old monsters. The bazaars glow under Spirit Lanterns, memory streams hold generations of knowledge, and the old riverfront mirrors it all. In Kal-shalà, past and future blur – a city where digital gods walk alongside ancient ones, and magic flows through circuits as easily as air."""
            ),
            ui.input_text_area(
                id="current_scene",
                label="Current Scene:",
                height="100px",
                value="""Tales of Unfathomable Power:

Tale 2: The Kal-Shalà of men

That night, a cold wind blew through the high beams in a forgotten skyscraper in the Old Financial District of the city. Abandoned for years, it had become a hub for lowlifes and a refuge for the destitute. Across the abandoned frame many tiny tents struggled against the cold and the winds. Inside one of them a couple argued, rising the volume of their voices bit by bit. Outside seated a young kid, looking absentmindedly at the distant city lights. There, covered by the blue haze the atmosphere draws on distant things rose the old, titanic, Temple, with all the buildings of the Government and its Protectorate latched to it like remoras or bloodsucking leeches. “Our Emperor,” the kid thought, Nowadays people disesteemed the old Emperor inside the Temple, calling it no more than an old archeological legacy from a distant time. Seldom anyone, even those that still had faith in the Vedanta, thought the ancient, unmoving Emperor was in any way still alive, but not his dad. “Epochs go by,” he had said to him, “cities crumble. The heavens change. But the Emperor remains living.” This had caused him a profound impression. Many years before, when he was still a little child, his father took him to see Him. One could still enter to the Temple at that time. He would forever remember the arid wind that blew through his face, as if conjured from nothing. The city had stopped being a desert many, many million years ago, when the continents were still joined as one, as his teachers had taught to him, and now was a humid place near the immense Ocean. And he remembered the old Emperor seated there, in an incredibly ancient, seemingly fragile throne of bones that seemed to cry, not from pain, but from pure sadness. And the light that shone on His head, holy and eternal. And he would forever remember his face. A face warm and distant, like an old father, looking at him. “Is... is he looking at me Dad?” He said to his father, and his father smiled “Yes, he sees all of Us.”

Dad was part of an increasingly radical group of Ninnuei fundamentalists. They argued these last millennia the pride of Man led to a stagnant society, where wealth and power was concentrated in the hands of the old nobility and the ever-putrid Party, its Protectorate and its Government. Forgetting about the Emperor and its old Visirs, prophets of old had made humanity morally rancid, and life unjust. Mom argued that he just blamed his own failures on external factors. That he did not think how we were poor, and he was weak. That we lived on a tent on the beams of an abandoned building.

In one moment, she said something the kid could not pretend not to hear

“I don’t know why I married you! You are a failure!”

Then silence came. Only the winds and the distant rumour of the city could be heard.

“I’m. I’m so sorry. I didn’t mean it.” Said the woman. Then the rustling of cloth could be heard as a young man with a dirty, unkempt beard and long, curly hair left the tent. The young kid recognized his own absentminded face in the face of his father. “Hi son. I... I’ll be back later, OK?” “Y... yes dad,” and for no reason he thought he needed to ask, “Want me to come with you?” His dad looked at him, confusion in his eyes, and doubted a second. “No, I’ll be fine. Take care of your mother, yes?” “Sure thing, dad.” “Well, bye.” “Bye... dad”

Soon his father disappeared in the shadows of the buildings. A couple of seconds after he was out of sight, his mother left the tent.

“Hey, Ji, have you seen your father?”

The kid looked at her with vacant eyes, and extended a finger.

“He went down, I think.”

“Didn’t he say where he was going?”

He shook his head negatively and ignored his mother, looking instead at the Temple far away. Various blimps, ships and flying constructs flew though the skies, and its distant lights drew lines in the immensity"""
            ),
            ui.input_numeric(
                "max_history",
                "Maximum Scene Pairs to Remember:",
                value=20,
                min=1,
                max=40,
                step=1,
            ),
            ui.row(
                ui.column(6, 
                    ui.input_action_button(
                        "new_game",
                        "New Game",
                        width="100%",
                        class_="btn-primary"
                    )
                ),
                ui.column(6,
                    ui.input_action_button(
                        "update_game",
                        "Update Game",
                        width="100%",
                        class_="btn-info"
                    )
                )
            ),
            ui.hr(),
            ui.markdown("""
            ### How to Use
            1. Select your preferred model provider and model
            2. Set your story world and plot
            3. Click "New Game" to start fresh or "Update Game" to apply changes
            4. Switch to the Chat tab
            5. Type your character's actions
            6. Watch as the story evolves - each response becomes the new current scene
            """)
        ),
        ui.nav_panel(
            "Chat",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.card(
                        ui.card_header("Save Management"),
                        ui.input_action_button(
                            "save_state",
                            "Save Current State",
                            width="100%",
                            class_="btn-primary mb-3"
                        ),
                        ui.input_select(
                            "save_select",
                            "Available Saves:",
                            choices=[],
                            width="100%"
                        ),
                        ui.input_action_button(
                            "load_save",
                            "Load Selected Save",
                            width="100%",
                            class_="btn-success mb-3"
                        ),
                        ui.input_action_button(
                            "regenerate",
                            "Regenerate Current Scene",
                            width="100%",
                            class_="btn-warning"
                        )
                    ),
                    ui.card(
                        ui.card_header("Save Information"),
                        ui.output_ui("save_info", fill=True)
                    ),
                    width=400
                ),
                ui.chat_ui("chat", placeholder="Enter your character's action...")
            )
        ),
        ui.nav_panel(
            "Story Elements",
            ui.output_text_verbatim("last_plan"),
            ui.markdown("""
            ### About Story Elements
            This tab shows the potential story elements that could naturally emerge 
            from the current scene. These elements guide the narrative responses 
            but aren't strictly followed, allowing for natural story development.
            """)
        ),
        ui.nav_panel(
            "Scene History",
            ui.output_text_verbatim("scene_history"),
            ui.markdown("""
            ### About Scene History
            This tab shows the chronological progression of scenes that are being 
            used to inform the narrative responses. Each entry represents a scene 
            that contributes to the story's continuity.
            """)
        ),
        selected="Chat"
    ),
    fillable_mobile=True
)

def update_save_list(adapter):
    """Update the save file choices in the UI"""
    saves = adapter.list_saves()
    choices = {save["path"]: save["display"] for save in saves}
    ui.update_select("save_select", choices=choices)

def server(input, output, session):
    logger.info("Server initialization started")
    
    # Create reactive values
    adapter_rv = reactive.Value()
    adapter_rv.set(WorkflowAdapter())
    logger.info("Created reactive workflow adapter")
    
    rv = reactive.Value()
    rv.set("No story elements generated yet. Start chatting to see potential story elements!")
    
    scenes_rv = reactive.Value([])

    # Update model choices when provider changes
    @reactive.Effect
    def _():
        provider = input.model_provider()
        if provider in MODELS_BY_PROVIDER:
            choices = MODELS_BY_PROVIDER[provider]
            ui.update_select("model_select", choices=dict(choices))
    
    # Initialize chat with only welcome message
    welcome_message = {
        "content": """Welcome to the Interactive Narrative! 
        Type your character's actions to see how the story unfolds.
        Each response will become your new current scene, building the narrative.""",
        "role": "assistant"
    }
    
    # Create chat instance with proper error handling
    chat = ui.Chat(
        "chat",
        messages=[welcome_message],  # Start with just the welcome message
        on_error="actual"  # Show actual errors for debugging
    )
    
    # Create chat controller
    controller = ChatController(input, chat, adapter_rv)
    
    # Initialize story state
    @reactive.Effect
    def _():
        adapter = adapter_rv.get()
        if not adapter.current_state:
            # For fresh start, add initial scene
            initial_scene = {
                "content": input.current_scene(),
                "role": "assistant"
            }
            adapter.create_initial_state(
                plot=input.plot(),
                current_scene=input.current_scene(),
                chat_messages=[welcome_message, initial_scene],
                scene_history=[]
            )
            # Add initial scene to chat
            @reactive.Effect
            async def _():
                await chat.append_message(initial_scene)
            logger.info("Initialized story state")
    
    # Transform assistant responses to handle markdown
    @chat.transform_assistant_response
    def _(response):
        return ui.markdown(response)
    
    @reactive.Effect
    @reactive.event(input.new_game)
    async def _():
        await controller.new_game()
    
    @reactive.Effect
    @reactive.event(input.update_game)
    async def _():
        await controller.update_game()
    
    @output
    @render.text
    def last_plan():
        return rv.get()
    
    @output
    @render.text
    def scene_history():
        history = scenes_rv.get()
        if not history:
            return "No previous scenes yet. Start chatting to build the story!"
        
        formatted_history = []
        for i, scene in enumerate(history, 1):
            formatted_history.append(f"Scene {i}:\n{scene}\n")
        
        return "\n".join(formatted_history)
    
    @output
    @render.ui
    def save_info():
        """Display information about the currently selected save."""
        selected_save = input.save_select()
        if not selected_save:
            return ui.markdown("No save selected")
            
        adapter = adapter_rv.get()
        metadata = adapter.metadata_adapter.load_metadata(os.path.join("saves", selected_save))
        
        if metadata:
            return ui.TagList(
                ui.markdown(f"### {metadata.story_name}"),
                ui.hr(),
                ui.markdown("#### Overall Summary"),
                ui.panel_well(metadata.overall_summary),
                ui.markdown("#### Latest Events"),
                ui.panel_well(metadata.latest_summary),
                ui.markdown(f"*Last Updated: {metadata.timestamp}*")
            )
        return ui.markdown("No metadata available for this save")
    
    # Initial save list population
    @reactive.Effect
    def _():
        update_save_list(adapter_rv.get())
    
    @reactive.Effect
    @reactive.event(input.save_state)
    async def _():
        await controller.save_state()
    
    @reactive.Effect
    @reactive.event(input.load_save)
    async def _():
        await controller.load_state(scenes_rv, rv)
    
    @reactive.Effect
    @reactive.event(input.regenerate)
    async def _():
        await controller.regenerate_scene(scenes_rv, rv)
    
    @chat.on_user_submit
    async def _():
        await controller.handle_user_action(scenes_rv, rv)

# Create and return the app
logger.info("Creating Shiny app")
app = App(app_ui, server)
logger.info("App creation complete")
