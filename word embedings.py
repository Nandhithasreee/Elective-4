import gensim.downloader as api
import numpy as np
import openai

# Add your API key here
openai.api_key = "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 "

# Load pre-trained word vectors
print("Loading word embedding model...")
model = api.load("glove-wiki-gigaword-50")
print("Model loaded successfully!")

# Function to find similar words
def get_similar_words(word):
    if word in model:
        similar_words = model.most_similar(word, topn=5)
        return [w[0] for w in similar_words]
    else:
        return []

# Function to generate response using OpenAI
def generate_response(prompt):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    return response.choices[0].message["content"]


# Original prompt
original_prompt = "Explain the importance of education."

# Find similar words for "education"
similar_words = get_similar_words("education")

# Create modified prompt using similar words
modified_prompt = original_prompt + " Also discuss " + ", ".join(similar_words)

# Generate responses
original_response = generate_response(original_prompt)
modified_response = generate_response(modified_prompt)

# Print results
print("\nOriginal Prompt:")
print(original_prompt)

print("\nModified Prompt:")
print(modified_prompt)

print("\nResponse for Original Prompt:")
print(original_response)


print("\nResponse for Modified Prompt:")
print(modified_response)