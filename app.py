import os
import logging
from shiny import App, ui, reactive, render
from app_utils import load_dotenv
from engine.workflow import generate_narrative

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('narrative_app')

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Define the UI
app_ui = ui.page_fillable(
    ui.panel_title("Interactive Narrative Chat"),
    ui.navset_tab(
        ui.nav_panel(
            "Story Settings",
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
                "Maximum Previous Scenes to Remember:",
                value=5,
                min=1,
                max=20,
                step=1
            ),
            ui.hr(),
            ui.markdown("""
            ### How to Use
            1. Set your story world and plot
            2. Switch to the Chat tab
            3. Type your character's actions
            4. Watch as the story evolves - each response becomes the new current scene
            """)
        ),
        ui.nav_panel(
            "Chat",
            ui.chat_ui("chat")
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
    
    # Create reactive value for storing last plan
    rv = reactive.Value()
    rv.set("No story elements generated yet. Start chatting to see potential story elements!")
    
    # Create reactive value for storing scene history
    scenes_rv = reactive.Value([])
    
    # Create chat instance with welcome message and initial scene
    welcome = {
        "content": """Welcome to the Interactive Narrative! 
        Type your character's actions to see how the story unfolds.
        Each response will become your new current scene, building the narrative.""",
        "role": "assistant"
    }
    initial_scene = {
        "content": "Standing at the entrance of a neon-lit dream parlor",
        "role": "assistant"
    }
    chat = ui.Chat("chat", messages=[welcome, initial_scene])
    logger.info("Chat interface initialized with welcome message")
    
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
    
    @chat.on_user_submit
    async def _():
        try:
            # Get the user's action
            user_action = chat.user_input()
            logger.info("Received user action: %s", user_action)
            
            # Add current scene to history before generating new one
            current_history = scenes_rv.get()
            current_history.append(input.current_scene())
            
            # Keep only the most recent scenes based on max_history setting
            max_history = int(input.max_history())
            if len(current_history) > max_history:
                logger.info("Trimming scene history to max length: %d", max_history)
                current_history = current_history[-max_history:]
            
            scenes_rv.set(current_history)
            logger.info("Updated scene history, current length: %d", len(current_history))
            
            # Generate narrative response
            logger.info("Generating narrative response")
            result = await generate_narrative(
                plot=input.plot(),
                current_scene=input.current_scene(),
                user_action=user_action,
                scene_history=scenes_rv.get()
            )
            
            if isinstance(result, dict):
                # Update last plan
                vision = result.get("original_vision", "")
                if vision:
                    logger.info("Updating story vision")
                    rv.set(vision)
                
                # Format and display the response
                narrative = result.get("narrative", "")
                if narrative:
                    logger.info("Updating current scene with new narrative")
                    # Update the current scene with the new narrative
                    ui.update_text_area("current_scene", value=narrative)
                    
                    await chat.append_message({
                        "content": narrative,
                        "role": "assistant"
                    })
                else:
                    logger.warning("Empty narrative response received")
                    await chat.append_message({
                        "content": "I couldn't generate a response. Please try again.",
                        "role": "assistant"
                    })
            else:
                # Handle string response
                logger.info("Processing string response")
                narrative = str(result)
                ui.update_text_area("current_scene", value=narrative)
                await chat.append_message({
                    "content": narrative,
                    "role": "assistant"
                })
                
        except Exception as e:
            logger.error("Error in chat handler: %s", str(e))
            await chat.append_message({
                "content": f"An error occurred: {str(e)}",
                "role": "assistant"
            })

# Create and return the app
logger.info("Creating Shiny app")
app = App(app_ui, server)
logger.info("App creation complete")
