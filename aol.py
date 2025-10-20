import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained chatbot model (e.g., HuggingFace DialoGPT model)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# CSV file for saving user data
USER_DATA_FILE = "user_data.csv"

# Function to save user data to a CSV file
def save_user_data(user_data):
    fieldnames = ["mood", "stress_level", "daily_activities", "sleep_hours"]
    try:
        # Check if the file already exists
        with open(USER_DATA_FILE, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header only if the file is empty
            file.seek(0)
            if file.read() == "":
                writer.writeheader()
            
            writer.writerow(user_data)
        print("User data saved successfully!")
    except Exception as e:
        print(f"Error saving user data: {e}")

# Function to load the latest user data from the CSV file
def load_latest_user_data():
    try:
        with open(USER_DATA_FILE, mode="r") as file:
            reader = csv.DictReader(file)
            data = list(reader)
            return data[-1] if data else None
    except FileNotFoundError:
        print("No user data file found.")
        return None

# Chatbot function
def chatbot():
    # Ask for user details
    print("Welcome to the Mental Health Chatbot!")
    mood = input("How are you feeling today? (e.g., happy, anxious): ")
    stress_level = input("What is your stress level? (low, medium, high): ")
    daily_activities = input("What were your main activities today? (e.g., work, exercise): ")
    sleep_hours = input("How many hours did you sleep last night?: ")

    # Save user data to CSV
    user_data = {
        "mood": mood,
        "stress_level": stress_level,
        "daily_activities": daily_activities,
        "sleep_hours": sleep_hours
    }
    save_user_data(user_data)

    # Start conversation
    print("\nGreat! Let's start our chat.")
    chat_history = ""

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Take care! Remember, seeking help is a strength.")
            break

        # Generate AI response
        latest_user_data = load_latest_user_data()
        intro = f"You are feeling {latest_user_data['mood']} and have a stress level of {latest_user_data['stress_level']}."
        input_text = intro + chat_history + f"\nUser: {user_input}\nAI:"
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=500, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)

        # Print AI response
        print(f"Chatbot: {response}")

        # Update chat history
        chat_history += f"\nUser: {user_input}\nAI: {response}"

# Run the chatbot
if __name__ == "__main__":
    chatbot()
