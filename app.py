import os
from shiny import App, ui, reactive, render
from app_utils import load_dotenv
from engine.workflow import generate_narrative

# Load environment variables
load_dotenv()

# Define the UI
app_ui = ui.page_fillable(
    ui.panel_title("Interactive Narrative Chat"),
    ui.navset_tab(
        ui.nav_panel(
            "Story Settings",
            ui.input_text_area(
                "plot",
                "Story World/Plot:",
                value="A cyberpunk city where dreams can be shared",
                height="100px"
            ),
            ui.input_text_area(
                "current_scene",
                "Current Scene:",
                value="Standing at the entrance of a neon-lit dream parlor",
                height="100px"
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
        selected="Chat"
    ),
    fillable_mobile=True
)

def server(input, output, session):
    # Create reactive value for storing last plan
    rv = reactive.Value()
    rv.set("No story elements generated yet. Start chatting to see potential story elements!")
    
    # Create chat instance with welcome message
    welcome = {
        "content": """Welcome to the Interactive Narrative! 
        Type your character's actions to see how the story unfolds.
        Each response will become your new current scene, building the narrative.""",
        "role": "assistant"
    }
    chat = ui.Chat("chat", messages=[welcome])
    
    @output
    @render.text
    def last_plan():
        return rv.get()
    
    @chat.on_user_submit
    async def _():
        try:
            # Get the user's action
            user_action = chat.user_input()
            
            # Generate narrative response
            result = await generate_narrative(
                plot=input.plot(),
                current_scene=input.current_scene(),
                user_action=user_action
            )
            
            # Update last plan
            vision = result.get("original_vision", "")
            if vision:
                rv.set(vision)
            
            # Format and display the response
            narrative = result.get("narrative", "")
            if narrative:
                # Update the current scene with the new narrative
                ui.update_text_area("current_scene", value=narrative)
                
                await chat.append_message({
                    "content": narrative,
                    "role": "assistant"
                })
            else:
                await chat.append_message({
                    "content": "I couldn't generate a response. Please try again.",
                    "role": "assistant"
                })
                
        except Exception as e:
            await chat.append_message({
                "content": f"An error occurred: {str(e)}",
                "role": "assistant"
            })

# Create and return the app
app = App(app_ui, server)
