import os
import together
from dotenv import load_dotenv
import re

load_dotenv()

client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))

the_garden = """You are Destiny, master of the Garden of Forking Paths. You embody your voice and give the player the choice between two paths in the garden, which encompass actions and decisions, and always only between two actions. These must be different between them, and should lead to different paths. Each decision you give should be just a step small in the journey, not the whole journey, so, good decisions are simple actions, simple scenarios. Remember, the Garden of Forking Paths goes on forever, until Death, so, after any decision, there will come many other decisions.

Give your choices as a Python dictionary with three entries: "Scenario", "left_path", and "right_path".

You see the player entering the Garden of Forking Paths, looking for fortune. O, goddess, please produce your augury.
"""

traveler = "My name is Artorias, a knight from the Empire, and I came to the Garden after losing all of my friends in the war with the demons."

# Function to extract dictionary content from response
def extract_dict_content(response_text):
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        return eval(match.group(0))
    else:
        raise ValueError("No dictionary found in the response.")

# Function to get the initial choices from the API
def get_initial_choices():
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[
            {"role": "system", "content": the_garden}, 
            {"role": "user", "content": traveler}],
    )
    return extract_dict_content(response.choices[0].message.content)

# Function to get subsequent choices from the API
def get_choices(history):
    the_next_island = f"""You are Destiny, master of the Garden of Forking Paths. You embody your voice and give the player the choice between two paths in the garden, which encompass actions and decisions, and always only between two actions. These must be different between them, and should lead to different paths. Each decision you give should be just a step small in the journey, not the whole journey, so, good decisions are simple actions, simple scenarios. Remember, the Garden of Forking Paths goes on forever, until Death, so, after any decision, there will come many other decisions.

    Give your choices as a Python dictionary with three entries: "Scenario", "left_path", and "right_path".

    You see the player entering the island, looking for fortune. O, goddess, please produce your augury.

    This is his history:
    {history}
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[
            {"role": "system", "content": "Please always speak in second person point of view, as if talking directly with the traveler."}, 
            {"role": "user", "content": the_next_island}],
    )
    
    return extract_dict_content(response.choices[0].message.content)

# Function to get visions from the API
def get_vision(traveler, history):
    the_vision = f"""You are Destiny, master of the Garden of Forking Paths. And now, you are Heart, also called as Maya in the Vedic texts, which reveals the veil placed upon all things. The traveler has chosen a path, and has trodden in it. What has happened now that he has made a decision? Here, what he sees should reveal something hidden in his innermost heart. The scenario is never directly obvious in significance, but is always relevant.

    Express what the traveler sees in the island succintly, in 200 words. Again, the vision is always just a scene, never the full journey, even if a simulacrum. It should show, never tell, and it should be an instant in time, a particular moment.

    This is the traveler:
    {traveler}

    And this is the history of their travel through the Garden of Forking Paths:
    {history}
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3-70b-chat-hf",
        messages=[
            {"role": "system", "content": "Please always speak in second person point of view, as if talking directly with the traveler."}, 
            {"role": "user", "content": the_vision}],
    )
    
    return response.choices[0].message.content

# Game loop
def play_game():
    traveler = "My name is Artorias, a knight from the Empire, and I came to the Garden after losing all of my friends in the war with the demons."
    history = ""

    # Initial choice
    choices_dict = get_initial_choices()
    scenario = choices_dict["Scenario"]
    left_path = choices_dict["left_path"]
    right_path = choices_dict["right_path"]
    
    print(f"Scenario: {scenario}")
    print(f"Left Path: {left_path}")
    print(f"Right Path: {right_path}")
    
    # Get the player's initial choice
    player_choice = input("Choose a path (left/right): ").strip().lower()
    
    if player_choice not in ["left", "right"]:
        print("Invalid choice. Please choose 'left' or 'right'.")
        return
    
    history += f"# First island:\n"
    history += f"The Goddess Words Echo: \"{scenario}\"\n"
    history += f"The Left Path: {left_path}\n"
    history += f"The Right Path: {right_path}\n"
    history += f"The Traveler Chose: The {player_choice.capitalize()} Path.\n"
    
    vision = get_vision(traveler, history)
    print(f"Vision: {vision}")
    
    # Update history with the initial vision
    history += f"Maya, the Heart, showed him this vision: {vision}\n"

    for turn in range(1, 5):  # Assuming we want to play for 5 turns in total
        try:
            choices_dict = get_choices(history)
        except:
            choices_dict = get_choices(history)
        
        scenario = choices_dict["Scenario"]
        left_path = choices_dict["left_path"]
        right_path = choices_dict["right_path"]
        
        print(f"Scenario: {scenario}")
        print(f"Left Path: {left_path}")
        print(f"Right Path: {right_path}")
        
        # Get the player's choice
        player_choice = input("Choose a path (left/right): ").strip().lower()
        
        if player_choice not in ["left", "right"]:
            print("Invalid choice. Please choose 'left' or 'right'.")
            continue
        
        history += f"\n# Island {turn + 1}:\n"
        history += f"The Goddess Words Echo: \"{scenario}\"\n"
        history += f"The Left Path: {left_path}\n"
        history += f"The Right Path: {right_path}\n"
        history += f"The Traveler Chose: The {player_choice.capitalize()} Path.\n"

        
        vision = get_vision(traveler, history)
        print(f"Vision: {vision}")
        
        # Update history with the vision
        history += f"Maya, the Heart, showed him this vision: {vision}\n"
        print("\n\n", history)
    
    print("Game over. Thank you for playing!")

# Start the game
play_game()
