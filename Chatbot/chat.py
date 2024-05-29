import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Govt"

context = {"scheme": None}  # Initialize context

def get_response(msg):
    global context
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Update context if user asks about a scheme
                if intent["tag"] == "government_scheme":
                    context["scheme"] = None
                elif intent["tag"] == "about":
                    for pattern in intent["patterns"]:
                        for word in sentence:
                            if word in pattern:
                                scheme = pattern.split("about")[-1].strip()
                                context["scheme"] = scheme
                                break
                elif intent["tag"] == "more_info":
                    scheme = context.get("scheme")
                    if scheme:
                        intent["patterns"] = [f"more information about {scheme}"]
                elif intent["tag"] == "eligibility_criteria":
                    scheme = context.get("scheme")
                    if scheme:
                        intent["patterns"] = [f"eligibility criteria for {scheme}"]
                elif intent["tag"] == "application_process":
                    scheme = context.get("scheme")
                    if scheme:
                        intent["patterns"] = [f"application process for {scheme}"]
                return random.choice(intent['responses'])
    
    return "Sorry! I do not understand...Please specify the scheme name correctly"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)