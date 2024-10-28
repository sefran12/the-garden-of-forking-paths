import os
import logging
import traceback
from shiny import App, ui, reactive, render
from app_utils import load_dotenv
from adapter.adapter import WorkflowAdapter
from resource_manager import ResourceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('narrative_app')

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Get the singleton instance of ResourceManager (already initialized in run.py)
resource_manager = ResourceManager()
logger.info(f"Using language: {resource_manager.get_current_language()}")

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
                "content": resource_manager.get_text("ui.welcome_message"),
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

def update_save_list(adapter):
    """Update the save file choices in the UI"""
    saves = adapter.list_saves()
    choices = {save["path"]: save["display"] for save in saves}
    ui.update_select("save_select", choices=choices)

# Define the UI
app_ui = ui.page_fillable(
    ui.panel_title(f"Interactive Narrative Chat - {resource_manager.get_current_language().upper()}"),
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
                value=resource_manager.get_text("default_content.plot")
            ),
            ui.input_text_area(
                id="current_scene",
                label="Current Scene:",
                height="100px",
                value=resource_manager.get_text("default_content.initial_scene")
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
            ui.markdown(resource_manager.get_text("ui.how_to_use"))
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
            ui.markdown(resource_manager.get_text("ui.about_story_elements"))
        ),
        ui.nav_panel(
            "Scene History",
            ui.output_text_verbatim("scene_history"),
            ui.markdown(resource_manager.get_text("ui.about_scene_history"))
        ),
        selected="Chat"
    ),
    fillable_mobile=True
)

def server(input, output, session):
    logger.info("Server initialization started")
    
    # Create reactive values
    adapter_rv = reactive.Value()
    adapter_rv.set(WorkflowAdapter())
    logger.info("Created reactive workflow adapter")
    
    rv = reactive.Value()
    rv.set(resource_manager.get_text("ui.no_story_elements"))
    
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
        "content": resource_manager.get_text("ui.welcome_message"),
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
            return resource_manager.get_text("ui.no_scene_history")
        
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
            return ui.markdown(resource_manager.get_text("ui.no_save_selected"))
            
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
        return ui.markdown(resource_manager.get_text("ui.no_metadata"))
    
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

# Create the app
app = App(app_ui, server)

# Only run the app if this file is run directly
if __name__ == "__main__":
    app.run()
