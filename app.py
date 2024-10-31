import os
import logging
import traceback
from shiny import App, ui, reactive, render
from app_utils import load_dotenv
from adapter.adapter import WorkflowAdapter
from ui import app_ui, MODELS_BY_PROVIDER
from pymongo import MongoClient
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('narrative_app')

# Load environment variables
load_dotenv(override=True)
logger.info("Environment variables loaded")

# MongoDB connection
mongo_client = MongoClient(os.getenv('MONGODB_URI'))
db = mongo_client[os.getenv('MONGODB_DB_NAME')]
saves_collection = db[os.getenv('MONGODB_SAVES_COLLECTION')]
metadata_collection = db[os.getenv('MONGODB_METADATA_COLLECTION')]
logger.info("MongoDB connection established")

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
    
    def update_save_list(self):
        """Update the save file choices in the UI"""
        adapter = self.adapter_rv.get()
        saves = adapter.list_saves()
        choices = {save["id"]: save["display"] for save in saves}
        ui.update_select("save_select", choices=choices)
            
    async def new_game(self):
        try:
            with ui.Progress(min=0, max=3) as p:
                p.set(value=0, message="Initializing new game...", 
                      detail="Setting up adapter...")
                
                adapter = self.adapter_rv.get()
                await self.chat.clear_messages()
                
                p.set(value=1, message="Creating welcome message...", 
                      detail="Preparing initial scene...")
                
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
                
                p.set(value=2, message="Creating initial state...", 
                      detail="Setting up game environment...")
                
                adapter.create_initial_state(
                    plot=self.input.plot(),
                    current_scene=self.input.current_scene(),
                    chat_messages=[welcome_message, initial_scene],
                    scene_history=[]
                )
                
                p.set(value=3, message="Game started successfully!")
            
            ui.notification_show("New game started successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to start new game: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def update_game(self):
        try:
            with ui.Progress(min=0, max=2) as p:
                p.set(value=0, message="Updating game state...", 
                      detail="Retrieving adapter...")
                
                adapter = self.adapter_rv.get()
                if not adapter.current_state:
                    ui.notification_show("No active game to update. Please start a new game first.", type="warning")
                    return
                    
                p.set(value=1, message="Applying updates...", 
                      detail="Updating plot and scene...")
                
                adapter.current_state.plot = self.input.plot()
                adapter.current_state.current_scene = self.input.current_scene()
                
                p.set(value=2, message="Game updated successfully!")
            
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
                
                p.set(value=1, message="Saving state...", 
                      detail="Saving to file and MongoDB...")
                
                save_path, save_id = await adapter.save_state(workflow_config=config)
                
                p.set(value=2, message="Finalizing save...", 
                      detail="Updating save list...")
                
                self.update_save_list()
                
                p.set(value=3, message="Save complete!")
            
            ui.notification_show(f"State saved successfully to file: {save_path} and MongoDB ID: {save_id}", type="message")
            
        except Exception as e:
            error_msg = f"Failed to save state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def load_state(self, scenes_rv, rv):
        try:
            with ui.Progress(min=0, max=4) as p:
                p.set(value=0, message="Loading saved state...", 
                      detail="Retrieving adapter...")
                
                adapter = self.adapter_rv.get()
                selected_save = self.input.save_select()
                if not selected_save:
                    return
                
                p.set(value=1, message="Loading state data...", 
                      detail="Reading from storage...")
                    
                state = adapter.load_state(selected_save)
                
                p.set(value=2, message="Updating UI...", 
                      detail="Applying loaded state...")
                
                ui.update_text_area("plot", value=state.plot)
                ui.update_text_area("current_scene", value=state.current_scene)
                scenes_rv.set(state.scene_history)
                
                p.set(value=3, message="Restoring chat history...", 
                      detail="Loading messages...")
                
                await self.chat.clear_messages()
                for msg in state.chat_messages:
                    await self.chat.append_message(msg)
                
                if "original_vision" in state.metadata:
                    rv.set(state.metadata["original_vision"])
                    
                p.set(value=4, message="State loaded successfully!")
            
            source = "local file" if adapter.current_save_path else "MongoDB"
            ui.notification_show(f"State loaded successfully from {source}", type="message")
            
        except Exception as e:
            error_msg = f"Failed to load state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
            
    async def regenerate_scene(self, scenes_rv, rv):
        try:
            with ui.Progress(min=0, max=4) as p:
                p.set(value=0, message="Preparing scene regeneration...", 
                      detail="Gathering messages...")
                
                adapter = self.adapter_rv.get()
                messages = [
                    {"content": msg["content"], "role": msg["role"]} 
                    for msg in self.chat.messages()[:-1]
                ]
                
                p.set(value=1, message="Configuring workflow...", 
                      detail="Setting up model parameters...")
                
                config = self.get_model_info()
                
                p.set(value=2, message="Regenerating scene...", 
                      detail="Processing with language model...")
                
                state = await adapter.regenerate_current_state(
                    chat_messages=messages,
                    max_scenes=int(self.input.max_history()),
                    workflow_config=config
                )
                
                p.set(value=3, message="Updating interface...", 
                      detail="Applying new scene...")
                
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
                
                p.set(value=4, message="Scene regenerated successfully!")
            
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
            with ui.Progress(min=0, max=4) as p:
                p.set(value=0, message="Processing user action...", 
                      detail="Initializing response...")
                
                adapter = self.adapter_rv.get()
                user_action = self.chat.user_input()
                logger.info("Received user action: %s", user_action)
                
                p.set(value=1, message="Updating scene history...", 
                      detail="Recording current scene...")
                
                current_history = scenes_rv.get()
                current_history.append(self.input.current_scene())
                scenes_rv.set(current_history)
                logger.info("Updated scene history, current length: %d", len(current_history))
                
                p.set(value=2, message="Generating response...", 
                      detail="Processing with language model...")
                
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
                
                p.set(value=3, message="Updating interface...", 
                      detail="Applying new scene...")
                
                ui.update_text_area("current_scene", value=state.current_scene)
                
                if "original_vision" in state.metadata:
                    rv.set(state.metadata["original_vision"])
                
                await self.chat.append_message({
                    "content": state.current_scene,
                    "role": "assistant"
                })
                
                p.set(value=4, message="Response generated successfully!")
                
        except Exception as e:
            error_msg = f"Error in chat handler: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            await self.chat.append_message({
                "content": f"Error: {error_msg}",
                "role": "assistant"
            })

def server(input, output, session):
    logger.info("Server initialization started")
    
    # Create reactive values
    adapter_rv = reactive.Value()
    adapter_rv.set(WorkflowAdapter())
    logger.info("Created reactive workflow adapter")
    
    rv = reactive.Value()
    rv.set("No story elements generated yet. Start chatting to see potential story elements!")
    
    scenes_rv = reactive.Value([])

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
    
    # Reactive effects
    @reactive.Effect
    def _():
        provider = input.model_provider()
        if provider in MODELS_BY_PROVIDER:
            choices = MODELS_BY_PROVIDER[provider]
            ui.update_select("model_select", choices=dict(choices))
    
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
    
    @reactive.Effect
    def _():
        controller.update_save_list()
    
    @reactive.Effect
    @reactive.event(input.new_game)
    async def _():
        await controller.new_game()
    
    @reactive.Effect
    @reactive.event(input.update_game)
    async def _():
        await controller.update_game()
    
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
    
    # Outputs
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
        try:
            state = adapter.load_state(selected_save)
            
            # Try to get metadata from MongoDB if it's a MongoDB save
            metadata = state.metadata
            if adapter.current_save_id:
                mongo_metadata = metadata_collection.find_one({"save_id": adapter.current_save_id})
                if mongo_metadata:
                    # Remove MongoDB-specific fields
                    mongo_metadata.pop('_id', None)
                    mongo_metadata.pop('save_id', None)
                    metadata.update(mongo_metadata)
            
            return ui.TagList(
                ui.markdown(f"### {metadata.get('story_name', 'Untitled')}"),
                ui.hr(),
                ui.markdown("#### Overall Summary"),
                ui.panel_well(metadata.get('overall_summary', 'No summary available')),
                ui.markdown("#### Latest Events"),
                ui.panel_well(metadata.get('latest_summary', 'No recent events')),
                ui.markdown(f"*Last Updated: {state.timestamp}*")
            )
        except Exception as e:
            logger.error(f"Failed to load save info: {str(e)}")
            return ui.markdown("Error loading save information")
    
    # Transform assistant responses to handle markdown
    @chat.transform_assistant_response
    def _(response):
        return ui.markdown(response)
    
    # Chat user submit handler
    @chat.on_user_submit
    async def _():
        await controller.handle_user_action(scenes_rv, rv)

# Create and return the app
logger.info("Creating Shiny app")
app = App(app_ui, server)
logger.info("App creation complete")
