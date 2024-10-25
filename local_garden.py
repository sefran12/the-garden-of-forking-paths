import os
import json
import ollama
from dotenv import load_dotenv
import re
import sqlite3
import random
import textwrap
from colorama import Fore, Style, init

load_dotenv()

# Initialize colorama
init()

LLM_MODEL = os.getenv("OLLAMA_MODEL")

the_garden = """You are Destiny, master of the Garden of Forking Paths. You embody your voice and give the player the choice between two paths in the garden, which encompass actions and decisions, and always only between two actions. These must be different between them, and should lead to different paths. Each decision you give should be just a step small in the journey, not the whole journey, so, good decisions are simple actions, simple scenarios. Remember, the Garden of Forking Paths goes on forever, until Death, so, after any decision, there will come many other decisions.

Give your choices as a JSON code block with three entries: "Scenario", "left_path", and "right_path". Enclose the JSON in backticks. Here are examples of good responses:

```json
{
    "Scenario": "You come across a fork in the path. To your left, a dense forest beckons with mysterious whispers. To your right, a steep mountain trail promises a challenging climb.",
    "left_path": "Enter the whispering forest",
    "right_path": "Climb the steep mountain trail"
}
```

You see the player entering the Garden of Forking Paths, looking for fortune. O, goddess, please produce your augury.
"""

traveler = "My name is Artorias, a knight from the Empire, and I came to the Garden after losing all of my friends in the war with the demons."

astral_cards = [
    "The Star of Hope: Brings a renewed sense of purpose and strength.",
    "The Shadow of Despair: Casts a dark cloud, leading to moments of doubt and fear.",
    "The Flame of Passion: Ignites a burning desire to achieve one's goals.",
    "The Whisper of Betrayal: Reveals hidden treachery from a trusted ally.",
    "The Embrace of Comfort: Offers solace and healing from past wounds.",
    "The Chill of Isolation: Enforces a sense of loneliness and separation from others.",
    "The Song of Triumph: Celebrates a recent victory and encourages further successes.",
    "The Echo of Regret: Haunts with memories of past mistakes.",
    "The Light of Guidance: Illuminates the path forward, providing clarity and direction.",
    "The Veil of Mystery: Shrouds the future in uncertainty, creating an air of suspense."
]

# Define database functions
def init_db():
    conn = sqlite3.connect('game_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY,
                    turn INTEGER,
                    scenario TEXT,
                    left_path TEXT,
                    right_path TEXT,
                    player_choice TEXT,
                    astral_card TEXT,
                    vision TEXT
                 )''')
    conn.commit()
    return conn

def save_turn(conn, turn, scenario, left_path, right_path, player_choice, astral_card, vision):
    c = conn.cursor()
    c.execute('''INSERT INTO history (turn, scenario, left_path, right_path, player_choice, astral_card, vision) 
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', (turn, scenario, left_path, right_path, player_choice, astral_card, vision))
    conn.commit()

def load_history(conn):
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY turn')
    return c.fetchall()

def extract_dict_content(response_text):
    # Find JSON content enclosed in backticks
    match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            # Parse the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the response.")
    else:
        raise ValueError("No JSON code block found in the response.")

# Function to get the initial choices from the API
def get_initial_choices():
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": the_garden}, 
            {"role": "user", "content": traveler}],
    )
    return extract_dict_content(response['message']['content'])

# Function to get subsequent choices from the API
def get_choices(history):
    the_next_island = f"""You are Destiny, master of the Garden of Forking Paths. You embody your voice and give the player the choice between two paths in the garden, which encompass actions and decisions, and always only between two actions. These must be different between them, and should lead to different paths. Each decision you give should be just a step small in the journey, not the whole journey, so, good decisions are simple actions, simple scenarios. Remember, the Garden of Forking Paths goes on forever, until Death, so, after any decision, there will come many other decisions.
    
    Give your choices as a JSON code block with three entries: "Scenario", "left_path", and "right_path". Enclose the JSON in backticks. Here are examples of good responses:

    ```json
    {{
        "Scenario": "You come across a fork in the path. To your left, a dense forest beckons with mysterious whispers. To your right, a steep mountain trail promises a challenging climb.",
        "left_path": "Enter the whispering forest",
        "right_path": "Climb the steep mountain trail"
    }}
    ```

    You see the player entering the island, looking for fortune. O, goddess, please produce your augury.

    This is his history:
    {history}
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Please always speak in second person point of view, as if talking directly with the traveler."},
            {"role": "user", "content": the_next_island}],
    )

    return extract_dict_content(response['message']['content'])

# Function to get visions from the API
def get_vision(traveler, history, astral_card):
    the_vision = f"""You are Destiny, master of the Garden of Forking Paths. And now, you are Heart, also called as Maya in the Vedic texts, which reveals the veil placed upon all things. The traveler has chosen a path, and has trodden in it. What has happened now that he has made a decision?

    Express what the traveler sees in the island succintly, in 200 words. Again, the vision is always just a scene, never the full journey, even if a simulacrum. It should show, never tell, and it should be an instant in time, a particular moment.

    This is the traveler:
    {traveler}

    This is their astral card, unknown to them, which affects their fortune. This knowledge is secret to you, and cannot be expressed to the traveller. It just influences their travel:
    {astral_card}

    And this is the history of their travel through the Garden of Forking Paths:
    {history}
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Please always speak in second person point of view, as if talking directly with the traveler."}, 
            {"role": "user", "content": the_vision}],
    )
    
    return response['message']['content']

# Function to print formatted text
def print_formatted(text, color=Fore.WHITE, width=120, justify=True):
    wrapper = textwrap.TextWrapper(width=width, break_long_words=True, replace_whitespace=False)
    lines = wrapper.wrap(text)
    for line in lines:
        if justify:
            print(color + line.ljust(width) + Style.RESET_ALL)
        else:
            print(color + line + Style.RESET_ALL)

# Game loop
def play_game():
    traveler = "My name is Artorias, a knight from the Empire, and I came to the Garden after losing all of my friends in the war with the demons."
    history = ""
    conn = init_db()  # Initialize the database
    turn = 0  # Initialize turn counter
    
    # Initial choice
    choices_dict = get_initial_choices()
    scenario = choices_dict["Scenario"]
    left_path = choices_dict["left_path"]
    right_path = choices_dict["right_path"]
        
    print_formatted(f"Scenario: {scenario}", Fore.CYAN)
    print_formatted(f"Left Path: {left_path}", Fore.GREEN)
    print_formatted(f"Right Path: {right_path}", Fore.YELLOW)
    print_formatted("Your Own Path: Create your own path", Fore.MAGENTA)
        
    # Get the player's initial choice
    player_choice = input(Fore.WHITE + "Choose a path (left/right/own): " + Style.RESET_ALL).strip().lower()
        
    if player_choice not in ["left", "right", "own"]:
        print_formatted("Invalid choice. Please choose 'left', 'right', or 'own'.", Fore.RED)
        return
    if player_choice == "own":
        own_path = input(Fore.WHITE + "Describe your own path: " + Style.RESET_ALL)
        player_choice_description = own_path
    else:
        player_choice_description = left_path if player_choice == "left" else right_path
        
    history += f"# First island:\n"
    history += f"The Goddess Words Echo: \"{scenario}\"\n"
    history += f"The Left Path: {left_path}\n"
    history += f"The Right Path: {right_path}\n"
    history += f"The Traveler Chose: {player_choice_description}\n"
    
    astral_card = random.choice(astral_cards)
    vision = get_vision(traveler, history, astral_card)
    print_formatted(f"This is your astral card: {astral_card}", Fore.BLUE)
    print_formatted(f"Vision: {vision}", Fore.CYAN)

    # Update history with the initial vision
    history += f"Maya, the Heart, showed him this vision: {vision}\n"
    save_turn(conn, turn, scenario, left_path, right_path, player_choice_description, astral_card, vision)
    
    for turn in range(1, 50):  # Assuming we want to play for 50 turns in total
        try:
            choices_dict = get_choices(history)
        except:
            choices_dict = get_choices(history)     
        
        scenario = choices_dict["Scenario"]
        left_path = choices_dict["left_path"]
        right_path = choices_dict["right_path"]
        
        print_formatted(f"\nScenario: {scenario}", Fore.CYAN)
        print_formatted(f"Left Path: {left_path}", Fore.GREEN)
        print_formatted(f"Right Path: {right_path}", Fore.YELLOW)
        print_formatted("Your Own Path: Create your own path", Fore.MAGENTA)
        
        # Get the player's choice
        player_choice = input(Fore.WHITE + "Choose a path (left/right/own): " + Style.RESET_ALL).strip().lower()
        
        if player_choice not in ["left", "right", "own"]:
            print_formatted("Invalid choice. Please choose 'left', 'right', or 'own'.", Fore.RED)
            continue
        
        if player_choice == "own":
            own_path = input(Fore.WHITE + "Describe your own path: " + Style.RESET_ALL)
            player_choice_description = own_path
        else:
            player_choice_description = left_path if player_choice == "left" else right_path
        
        history += f"\n# Island {turn + 1}:\n"
        history += f"The Goddess Words Echo: \"{scenario}\"\n"
        history += f"The Left Path: {left_path}\n"
        history += f"The Right Path: {right_path}\n"
        history += f"The Traveler Chose: {player_choice_description}\n"
        
        astral_card = random.choice(astral_cards)
        vision = get_vision(traveler, history, astral_card)
        print_formatted(f"This is your astral card: {astral_card}", Fore.BLUE)
        print_formatted(f"Vision: {vision}", Fore.CYAN)
        # Update history with the vision
        history += f"Maya, the Heart, showed him this vision: {vision}\n"
        save_turn(conn, turn, scenario, left_path, right_path, player_choice, astral_card, vision)
    print_formatted("Game over. Thank you for playing!", Fore.GREEN)

# Start the game
play_game()
