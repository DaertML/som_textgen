# Importing required libraries
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Define some sentences for training
sentences = [
    "I love machine learning.",
    "Python is an amazing language.",
    "Self organizing maps are useful.",
    "Text generation is interesting.",
    "Neural networks can learn patterns.",
    "Coding is fun.",
    "Artificial intelligence is the future.",
]

# Preprocess the sentences into word vectors
def preprocess(sentences):
    # Tokenizing the sentences into words
    words = [sentence.lower().split() for sentence in sentences]
    
    # Flatten the list of words
    words = [word for sentence in words for word in sentence]
    
    # Encoding words using Label Encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(words)
    
    # One-Hot Encoding
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    # Create a word to vector mapping
    word_to_vec = {word: onehot_encoded[i] for i, word in enumerate(label_encoder.classes_)}
    vec_to_word = {tuple(onehot_encoded[i]): word for i, word in enumerate(label_encoder.classes_)}
    
    # Convert sentences into vectors
    vector_sentences = [[word_to_vec[word] for word in sentence.lower().split()] for sentence in sentences]
    
    return vector_sentences, word_to_vec, vec_to_word

# Preprocessing sentences
vector_sentences, word_to_vec, vec_to_word = preprocess(sentences)

# Flattening the vector sentences to train the SOM
data = np.array([vector for sentence in vector_sentences for vector in sentence])

# Initialize and train the SOM
som_size = (100, 100)
som = MiniSom(som_size[0], som_size[1], len(data[0]), sigma=0.3, learning_rate=0.5)
som.random_weights_init(data)
som.train_random(data, 100)

# Function to generate text using SOM
def generate_text(som, word_to_vec, vec_to_word, start_word, num_words):
    current_word = start_word.lower()  # Convert start word to lowercase
    result = [current_word]
    
    for _ in range(num_words):
        # Get the vector of the current word
        current_vec = word_to_vec.get(current_word, None)
        if current_vec is None:
            break

        # Find the BMU (Best Matching Unit) in the SOM for the current vector
        bmu = som.winner(current_vec)
        bmu_vec = som.get_weights()[bmu]
        
        # Find the closest word to the BMU vector
        closest_word = min(vec_to_word.keys(), key=lambda k: np.linalg.norm(np.array(k) - bmu_vec))
        
        # Add the closest word to the result
        next_word = vec_to_word.get(tuple(closest_word), None)
        if next_word is None:
            break
        
        result.append(next_word)
        current_word = next_word
    
    return ' '.join(result)

# Test text generation
start_word = "I love"  # Starting word for text generation
generated_text = generate_text(som, word_to_vec, vec_to_word, start_word, 100)
print("Generated text:", generated_text)

