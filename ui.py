from shiny import ui

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
    ("actor-critic", "Actor-Critic - Policy-based narrative generation"),
    ("policy-gradient-actor-critic", "Policy Gradient A2C - Gradient update, Policy-based narrative generation"),
    ("dimensional-critic", "Dimensional Critic - Multi-dimensional narrative analysis"),
    ("selective-critic", "Selective Critic - Context-aware actor selection"),
    ("optimizing-critic", "Optimizing Critic - Direct narrative optimization")
]

# Group models by provider for UI
MODELS_BY_PROVIDER = {}
for model in AVAILABLE_MODELS:
    provider = model["provider"]
    if provider not in MODELS_BY_PROVIDER:
        MODELS_BY_PROVIDER[provider] = []
    display_name = f"{model['name']} ({model['size']})" if model['size'] != "N/A" else model['name']
    MODELS_BY_PROVIDER[provider].append((model['name'], display_name))

# Define the UI with improved styling and layout
app_ui = ui.page_fillable(
    ui.tags.style("""
        .nav-tabs { margin-bottom: 20px; }
        .card { margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        .btn { margin-bottom: 10px; }
        .well { background-color: #f8f9fa; padding: 15px; border-radius: 4px; }
    """),
    ui.div(
        ui.h2("Interactive Narrative Chat"),
        style="margin-bottom: 20px;"
    ),
    ui.navset_tab(
        ui.nav_panel(
            "Story Settings",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.card(
                        ui.h4("Model Configuration"),
                        ui.input_select(
                            "workflow_type",
                            "Narrative Engine:",
                            choices=dict(WORKFLOW_TYPES),
                            width="100%"
                        ),
                        ui.input_select(
                            "model_provider",
                            "Provider:",
                            choices=list(MODELS_BY_PROVIDER.keys()),
                            width="100%"
                        ),
                        ui.input_select(
                            "model_select",
                            "Language Model:",
                            choices=[],
                            width="100%"
                        ),
                        ui.input_numeric(
                            "max_history",
                            "Maximum Scene Memory:",
                            value=20,
                            min=1,
                            max=40,
                            step=1,
                            width="100%"
                        ),
                    ),
                    width=300
                ),
                ui.card(
                    ui.h4("Story Configuration"),
                    ui.input_text_area(
                        id="plot",
                        label="Story World/Plot:",
                        height="200px",
                        width="100%",
                        value="""The city-state of Kal-shalà stands in the shadow of the ancient Nameless Emperor's throne. Here, magic and technology merge – cybernetic skyscrapers rise beside the Emperor's Palace-Temple, all turning with the Gearwheel of Fate.
The streets hold both old and new. The Ninnuei artifacts and Rusty Cauldron relics share space with AI programs and augmented citizens. In the hyper-district, holographic ads light the night, while beneath the streets, the Spirit Network carries encrypted data through the city's foundations.
The ruling Dynasty, descended from the Emperor's Court, governs through a mix of ancient rites and modern methods. Among the citizens walk street urchins who shape the city's destiny, cyborg oracles reading futures in holograms, technomancers coding spells, and scholars piecing together forgotten histories.
Children play with digital ghosts of ancient beasts while rogue AIs take the forms of old monsters. The bazaars glow under Spirit Lanterns, memory streams hold generations of knowledge, and the old riverfront mirrors it all. In Kal-shalà, past and future blur – a city where digital gods walk alongside ancient ones, and magic flows through circuits as easily as air."""
                    ),
                    ui.input_text_area(
                        id="current_scene",
                        label="Current Scene:",
                        height="200px",
                        width="100%",
                        value="""Tales of Unfathomable Power:

Tale 2: The Kal-Shalà of men

That night, a cold wind blew through the high beams in a forgotten skyscraper in the Old Financial District of the city. Abandoned for years, it had become a hub for lowlifes and a refuge for the destitute. Across the abandoned frame many tiny tents struggled against the cold and the winds. Inside one of them a couple argued, rising the volume of their voices bit by bit. Outside seated a young kid, looking absentmindedly at the distant city lights. There, covered by the blue haze the atmosphere draws on distant things rose the old, titanic, Temple, with all the buildings of the Government and its Protectorate latched to it like remoras or bloodsucking leeches. "Our Emperor," the kid thought, Nowadays people disesteemed the old Emperor inside the Temple, calling it no more than an old archeological legacy from a distant time. Seldom anyone, even those that still had faith in the Vedanta, thought the ancient, unmoving Emperor was in any way still alive, but not his dad. "Epochs go by," he had said to him, "cities crumble. The heavens change. But the Emperor remains living." This had caused him a profound impression. Many years before, when he was still a little child, his father took him to see Him. One could still enter to the Temple at that time. He would forever remember the arid wind that blew through his face, as if conjured from nothing. The city had stopped being a desert many, many million years ago, when the continents were still joined as one, as his teachers had taught to him, and now was a humid place near the immense Ocean. And he remembered the old Emperor seated there, in an incredibly ancient, seemingly fragile throne of bones that seemed to cry, not from pain, but from pure sadness. And the light that shone on His head, holy and eternal. And he would forever remember his face. A face warm and distant, like an old father, looking at him. "Is... is he looking at me Dad?" He said to his father, and his father smiled "Yes, he sees all of Us."

Dad was part of an increasingly radical group of Ninnuei fundamentalists. They argued these last millennia the pride of Man led to a stagnant society, where wealth and power was concentrated in the hands of the old nobility and the ever-putrid Party, its Protectorate and its Government. Forgetting about the Emperor and its old Visirs, prophets of old had made humanity morally rancid, and life unjust. Mom argued that he just blamed his own failures on external factors. That he did not think how we were poor, and he was weak. That we lived on a tent on the beams of an abandoned building.

In one moment, she said something the kid could not pretend not to hear

"I don't know why I married you! You are a failure!"

Then silence came. Only the winds and the distant rumour of the city could be heard.

"I'm. I'm so sorry. I didn't mean it." Said the woman. Then the rustling of cloth could be heard as a young man with a dirty, unkempt beard and long, curly hair left the tent. The young kid recognized his own absentminded face in the face of his father. "Hi son. I... I'll be back later, OK?" "Y... yes dad," and for no reason he thought he needed to ask, "Want me to come with you?" His dad looked at him, confusion in his eyes, and doubted a second. "No, I'll be fine. Take care of your mother, yes?" "Sure thing, dad." "Well, bye." "Bye... dad"

Soon his father disappeared in the shadows of the buildings. A couple of seconds after he was out of sight, his mother left the tent.

"Hey, Ji, have you seen your father?"

The kid looked at her with vacant eyes, and extended a finger.

"He went down, I think."

"Didn't he say where he was going?"

He shook his head negatively and ignored his mother, looking instead at the Temple far away. Various blimps, ships and flying constructs flew though the skies, and its distant lights drew lines in the immensity"""
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
                ),
                ui.card(
                    ui.h4("How to Use"),
                    ui.markdown("""
                    1. Select your preferred model provider and model
                    2. Set your story world and plot
                    3. Click "New Game" to start fresh or "Update Game" to apply changes
                    4. Switch to the Chat tab
                    5. Type your character's actions
                    6. Watch as the story evolves - each response becomes the new current scene
                    """)
                )
            )
        ),
        ui.nav_panel(
            "Chat",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.card(
                        ui.h4("Save Management"),
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
                        ui.h4("Save Information"),
                        ui.output_ui("save_info", fill=True)
                    ),
                    width=350
                ),
                ui.chat_ui(
                    "chat",
                    placeholder="Enter your character's action...",
                    height="calc(100vh - 100px)"
                )
            )
        ),
        ui.nav_panel(
            "Story Elements",
            ui.card(
                ui.h4("Potential Story Elements"),
                ui.output_text_verbatim("last_plan"),
                ui.markdown("""
                ### About Story Elements
                This tab shows the potential story elements that could naturally emerge 
                from the current scene. These elements guide the narrative responses 
                but aren't strictly followed, allowing for natural story development.
                """)
            )
        ),
        ui.nav_panel(
            "Scene History",
            ui.card(
                ui.h4("Scene Progression"),
                ui.output_text_verbatim("scene_history"),
                ui.markdown("""
                ### About Scene History
                This tab shows the chronological progression of scenes that are being 
                used to inform the narrative responses. Each entry represents a scene 
                that contributes to the story's continuity.
                """)
            )
        ),
        selected="Chat"
    ),
    fillable_mobile=True
)
