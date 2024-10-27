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

# Group models by provider for UI
MODELS_BY_PROVIDER = {}
for model in AVAILABLE_MODELS:
    provider = model["provider"]
    if provider not in MODELS_BY_PROVIDER:
        MODELS_BY_PROVIDER[provider] = []
    display_name = f"{model['name']} ({model['size']})" if model['size'] != "N/A" else model['name']
    MODELS_BY_PROVIDER[provider].append((model['name'], display_name))

# Define the UI
app_ui = ui.page_fillable(
    ui.panel_title("Interactive Narrative Chat"),
    ui.navset_tab(
        ui.nav_panel(
            "Story Settings",
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
                "plot",
                "Story World/Plot:",
                value="""The city-state of Kal-shalà stands in the shadow of the ancient Nameless Emperor's throne. Here, magic and technology merge – cybernetic skyscrapers rise beside the Emperor's Palace-Temple, all turning with the Gearwheel of Fate.
The streets hold both old and new. The Ninnuei artifacts and Rusty Cauldron relics share space with AI programs and augmented citizens. In the hyper-district, holographic ads light the night, while beneath the streets, the Spirit Network carries encrypted data through the city's foundations.
The ruling Dynasty, descended from the Emperor's Court, governs through a mix of ancient rites and modern methods. Among the citizens walk street urchins who shape the city's destiny, cyborg oracles reading futures in holograms, technomancers coding spells, and scholars piecing together forgotten histories.
Children play with digital ghosts of ancient beasts while rogue AIs take the forms of old monsters. The bazaars glow under Spirit Lanterns, memory streams hold generations of knowledge, and the old riverfront mirrors it all. In Kal-shalà, past and future blur – a city where digital gods walk alongside ancient ones, and magic flows through circuits as easily as air.""",
                height="100px"
            ),
            ui.input_text_area(
                "current_scene",
                "Current Scene:",
                value="Standing at the entrance of a neon-lit dream parlor",
                height="100px"
            ),
            ui.input_numeric(
                "max_history",
                "Maximum Scene Pairs to Remember:",
                value=5,
                min=1,
                max=20,
                step=1,
            ),
            ui.hr(),
            ui.markdown("""
            ### How to Use
            1. Select your preferred model provider and model
            2. Set your story world and plot
            3. Switch to the Chat tab
            4. Type your character's actions
            5. Watch as the story evolves - each response becomes the new current scene
            """)
        ),
        ui.nav_panel(
            "Chat",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "save_select",
                        "Load Previous Save:",
                        choices=[]
                    ),
                    ui.input_action_button(
                        "load_save",
                        "Load Selected Save"
                    ),
                    ui.input_action_button(
                        "save_state",
                        "Save Current State"
                    ),
                    ui.input_action_button(
                        "regenerate",
                        "Regenerate Current Scene"
                    )
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
    
    # Get selected model info
    def get_model_info():
        provider = input.model_provider()
        model_name = input.model_select()
        return {
            "provider": provider,
            "model": model_name
        }
    
    # Initialize chat with welcome message
    welcome_message = {
        "content": """Welcome to the Interactive Narrative! 
        Type your character's actions to see how the story unfolds.
        Each response will become your new current scene, building the narrative.""",
        "role": "assistant"
    }
    initial_scene = {
        "content": "Standing at the entrance of a neon-lit dream parlor",
        "role": "assistant"
    }
    
    # Create chat instance with proper error handling
    chat = ui.Chat(
        "chat",
        messages=[welcome_message, initial_scene],
        on_error="actual"  # Show actual errors for debugging
    )
    
    # Transform assistant responses to handle markdown
    @chat.transform_assistant_response
    def _(response):
        # Return response as markdown
        return ui.markdown(response)
    
    @reactive.Effect
    def _():
        # Initialize story state if needed
        adapter = adapter_rv.get()
        if not adapter.current_state:
            adapter.create_initial_state(
                plot=input.plot(),
                current_scene=input.current_scene(),
                chat_messages=[welcome_message, initial_scene],
                scene_history=[]
            )
            logger.info("Initialized story state")
    
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
    
    def update_save_list():
        """Update the save file choices in the UI"""
        adapter = adapter_rv.get()
        saves = adapter.list_saves()
        ui.update_select("save_select", choices=saves)
    
    # Initial save list population
    @reactive.Effect
    def _():
        update_save_list()
    
    @reactive.Effect
    @reactive.event(input.save_state)
    def _():
        try:
            adapter = adapter_rv.get()
            # Extract just the messages from the chat
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in chat.messages()
            ]
            # Update current state with latest messages before saving
            adapter.current_state.chat_messages = messages
            save_path = adapter.save_state()
            # Update save list immediately after saving
            update_save_list()
            ui.notification_show(f"State saved to {save_path}", type="message")
        except Exception as e:
            error_msg = f"Failed to save state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
    
    @reactive.Effect
    @reactive.event(input.load_save)
    async def _():
        try:
            adapter = adapter_rv.get()
            selected_save = input.save_select()
            if not selected_save:
                return
                
            state = adapter.load_state(os.path.join("saves", selected_save))
            
            # Update UI with loaded state
            ui.update_text_area("plot", value=state.plot)
            ui.update_text_area("current_scene", value=state.current_scene)
            scenes_rv.set(state.scene_history)
            
            # Clear existing messages and load saved ones
            await chat.clear_messages()
            for msg in state.chat_messages:
                await chat.append_message(msg)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
                
            ui.notification_show("State loaded successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to load state: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            ui.notification_show(error_msg, type="error")
    
    @reactive.Effect
    @reactive.event(input.regenerate)
    async def _():
        try:
            adapter = adapter_rv.get()
            # Get current messages and remove the last assistant message
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in chat.messages()[:-1]  # Exclude last message
            ]
            
            # Pass selected model in workflow config
            config = get_model_info()
            state = await adapter.regenerate_current_state(
                chat_messages=messages,
                max_scenes=int(input.max_history()),
                workflow_config=config
            )
            
            # Update UI with regenerated state
            ui.update_text_area("current_scene", value=state.current_scene)
            scenes_rv.set(state.scene_history)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
                
            # Clear existing messages and reload them without the last one
            await chat.clear_messages()
            for msg in messages:
                await chat.append_message(msg)
            
            # Append the regenerated response
            await chat.append_message({
                "content": state.current_scene,
                "role": "assistant"
            })
            
            ui.notification_show("Scene regenerated successfully", type="message")
            
        except Exception as e:
            error_msg = f"Failed to regenerate scene: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            await chat.append_message({
                "content": f"Error: {error_msg}",
                "role": "assistant"
            })
    
    @chat.on_user_submit
    async def _():
        try:
            adapter = adapter_rv.get()
            
            # Get the user's action from chat
            user_action = chat.user_input()
            logger.info("Received user action: %s", user_action)
            
            # Add current scene to history before generating new one
            current_history = scenes_rv.get()
            current_history.append(input.current_scene())
            scenes_rv.set(current_history)
            logger.info("Updated scene history, current length: %d", len(current_history))
            
            # Extract just the messages from the chat
            messages = [
                {"content": msg["content"], "role": msg["role"]} 
                for msg in chat.messages()
            ]
            
            # Pass selected model in workflow config
            config = get_model_info()
            
            # Generate next state using adapter
            state = await adapter.generate_next_state(
                user_action=user_action,
                chat_messages=messages,
                max_scenes=int(input.max_history()),
                workflow_config=config
            )
            
            # Update UI with new state
            ui.update_text_area("current_scene", value=state.current_scene)
            
            if "original_vision" in state.metadata:
                rv.set(state.metadata["original_vision"])
            
            # Append assistant response
            await chat.append_message({
                "content": state.current_scene,
                "role": "assistant"
            })
                
        except Exception as e:
            error_msg = f"Error in chat handler: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_msg)
            await chat.append_message({
                "content": f"Error: {error_msg}",
                "role": "assistant"
            })

# Create and return the app
logger.info("Creating Shiny app")
app = App(app_ui, server)
logger.info("App creation complete")
