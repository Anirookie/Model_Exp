import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
import torch
import faiss
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from streamlit.web.cli import main as st_main
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Hugging Face model and tokenizer
model_name = "Ani8Face/Category"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Load the embedding model
embedding_model_name = "bert-base-uncased"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Define the combined model class
class CombinedModel(nn.Module):
    def __init__(self, config, num_labels_category, num_labels_subcategory):
        super(CombinedModel, self).__init__()
        self.bert = model  # Load the pre-trained model
        self.fc1 = nn.Linear(config.hidden_size + 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.category_classifier = nn.Linear(64, num_labels_category)
        self.subcategory_classifier = nn.Linear(64, num_labels_subcategory)

    def forward(self, input_ids, attention_mask, token_type_ids, amount):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_token_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
        combined_input = torch.cat((cls_token_output, amount.unsqueeze(1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        category_output = self.category_classifier(x)
        subcategory_output = self.subcategory_classifier(x)
        return category_output, subcategory_output

num_labels_category = 5  # Update with your actual number of categories
num_labels_subcategory = 5  # Update with your actual number of subcategories

# Initialize the model
model = CombinedModel(config=config, num_labels_category=num_labels_category, num_labels_subcategory=num_labels_subcategory)

# Initialize label encoders and scaler
category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()

# Load the Faiss index and descriptions
index = faiss.read_index("descriptions_index.faiss")
descriptions = np.load("descriptions.npy")

def get_embeddings(texts):
    inputs = embedding_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings

def retrieve_documents(description, top_k=3):
    # Convert the input description to an embedding
    description_embedding = get_embeddings([description])

    # Search the index for the top_k most similar embeddings
    distances, indices = index.search(description_embedding, top_k)

    # Retrieve the corresponding descriptions
    retrieved_docs = [descriptions[idx] for idx in indices[0]]
    return retrieved_docs

# Function to interact with the Google Generative AI API
def call_gemini_api(description, amount, all_amounts):
    try:
        # Retrieve relevant documents
        retrieved_docs = retrieve_documents(description)
        retrieved_docs_text = " ".join(retrieved_docs)

        # Preprocess the input
        inputs = tokenizer(description + " " + retrieved_docs_text, padding=True, truncation=True, return_tensors='pt')
        
        # Ensure all_amounts is a 2D array
        all_amounts_reshaped = [[amt] for amt in all_amounts]
        scaler = StandardScaler()
        scaler.fit(all_amounts_reshaped)
        
        # Ensure amount is a 2D array
        amount_scaled = scaler.transform([[amount]])

        # Convert amount_scaled to a string representation
        amount_scaled_str = str(amount_scaled[0][0])

        # Use a compatible model for text generation
        prompt = f"Transaction description: {description}, Retrieved docs: {retrieved_docs_text}, Amount: {amount_scaled_str}. Provide the category (e.g., Assets, Expenditure, Liabilities) and subcategory."

        # Generative AI API call
        response = genai.generate_text(prompt=prompt, model="models/text-bison-001")

        # Extract the generated text from the response
        generated_text = response.result

        # Assuming response contains the category and subcategory separated by a comma
        return generated_text
    except Exception as e:
        st.error(f"Error in calling Gemini API: {e}")
        return "Unknown, Unknown"

# Function to aggregate amounts by category
def update_balance_sheet(data):
    balance_sheet = data.groupby(['Predicted Category', 'Predicted Subcategory']).agg({'Amount': 'sum'}).reset_index()
    return balance_sheet

# Streamlit app interface
st.title("Financial Transaction Classifier with RAG")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Uploaded Data:")
    st.write(data)

    # Ensure columns exist and handle missing data
    if 'Description' in data.columns and 'Amount' in data.columns:
        descriptions = data['Description'].tolist()
        amounts = data['Amount'].tolist()

        # Fit and transform the scaler with the current amounts
        all_amounts = [amt for amt in amounts]
        
        # Get predictions for the entire dataset
        predictions = [call_gemini_api(desc, amt, all_amounts) for desc, amt in zip(descriptions, amounts)]

        # Debugging output for predictions
        st.write("Predictions:")
        st.write(predictions)

        # Handle unexpected responses
        categories = []
        subcategories = []
        for pred in predictions:
            try:
                category, subcategory = pred.split(',')
                categories.append(category.strip())
                subcategories.append(subcategory.strip())
            except ValueError:
                categories.append("Unknown")
                subcategories.append("Unknown")

        data['Predicted Category'] = categories
        data['Predicted Subcategory'] = subcategories

        st.write("Data with Predictions:")
        st.write(data)
        
        # Update and display balance sheet
        balance_sheet = update_balance_sheet(data)

        # Formatting the balance sheet
        st.write("Balance Sheet:")
        for category in balance_sheet['Predicted Category'].unique():
            st.write(f"### {category}")
            category_data = balance_sheet[balance_sheet['Predicted Category'] == category]
            for _, row in category_data.iterrows():
                st.write(f"{row['Predicted Subcategory']}: ${row['Amount']:,.2f}")
            total = category_data['Amount'].sum()
            st.write(f"**Total {category}: ${total:,.2f}**")

# User input prompt
description_input = st.text_input("Enter a transaction description")
amount_input = st.number_input("Enter the transaction amount", min_value=0.0, format="%.2f")

if st.button("Predict Category and Subcategory"):
    if description_input and amount_input:
        # Fit and transform with a single amount
        all_amounts = [amount_input]
        pred = call_gemini_api(description_input, amount_input, all_amounts)
        try:
            category, subcategory = pred.split(',')
            st.write(f"Predicted Category: {category.strip()}")
            st.write(f"Predicted Subcategory: {subcategory.strip()}")
        except ValueError:
            st.write("Prediction error: Unable to parse the response")

if __name__ == '__main__':
    st_main()


# import streamlit as st
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# import torch
# from torch import nn
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from streamlit.web.cli import main as st_main
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Load Hugging Face model and tokenizer
# model_name = "Ani8Face/Category"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# config = AutoConfig.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# # Define the model class
# class CombinedModel(nn.Module):
#     def __init__(self, config, num_labels_category, num_labels_subcategory):
#         super(CombinedModel, self).__init__()
#         self.bert = model  # Load the pre-trained model
#         self.fc1 = nn.Linear(config.hidden_size + 1, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.category_classifier = nn.Linear(64, num_labels_category)
#         self.subcategory_classifier = nn.Linear(64, num_labels_subcategory)

#     def forward(self, input_ids, attention_mask, token_type_ids, amount):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         cls_token_output = outputs.last_hidden_state[:, 0, :]  # CLS token output
#         combined_input = torch.cat((cls_token_output, amount.unsqueeze(1)), dim=1)
#         x = torch.relu(self.fc1(combined_input))
#         x = torch.relu(self.fc2(x))
#         category_output = self.category_classifier(x)
#         subcategory_output = self.subcategory_classifier(x)
#         return category_output, subcategory_output

# num_labels_category = 5  # Update with your actual number of categories
# num_labels_subcategory = 5  # Update with your actual number of subcategories

# # Initialize the model
# model = CombinedModel(config=config, num_labels_category=num_labels_category, num_labels_subcategory=num_labels_subcategory)

# # Initialize label encoders and scaler
# category_encoder = LabelEncoder()
# subcategory_encoder = LabelEncoder()

# # Define function to interact with the Google Generative AI API
# def call_gemini_api(description, amount, all_amounts):
#     try:
#         # Preprocess the input
#         inputs = tokenizer(description, padding=True, truncation=True, return_tensors='pt')
        
#         # Ensure all_amounts is a 2D array
#         all_amounts_reshaped = [[amt] for amt in all_amounts]
#         scaler = StandardScaler()
#         scaler.fit(all_amounts_reshaped)
        
#         # Ensure amount is a 2D array
#         amount_scaled = scaler.transform([[amount]])

#         # Convert amount_scaled to a string representation
#         amount_scaled_str = str(amount_scaled[0][0])

#         # Use a compatible model for text generation
#         prompt = f"Transaction description: {description}, Amount: {amount_scaled_str}. Provide the category (e.g., Assets, Expenditure, Liabilities) and subcategory."

#         # Generative AI API call
#         response = genai.generate_text(prompt=prompt, model="models/text-bison-001")

#         # Extract the generated text from the response
#         generated_text = response.result

#         # Assuming response contains the category and subcategory separated by a comma
#         return generated_text
#     except Exception as e:
#         st.error(f"Error in calling Gemini API: {e}")
#         return "Unknown, Unknown"

# # Function to aggregate amounts by category
# def update_balance_sheet(data):
#     balance_sheet = data.groupby(['Predicted Category', 'Predicted Subcategory']).agg({'Amount': 'sum'}).reset_index()
#     return balance_sheet

# # Streamlit app interface
# st.title("Financial Transaction Classifier")

# # File upload
# uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         data = pd.read_csv(uploaded_file)
#     else:
#         data = pd.read_excel(uploaded_file)
    
#     st.write("Uploaded Data:")
#     st.write(data)

#     # Ensure columns exist and handle missing data
#     if 'Description' in data.columns and 'Amount' in data.columns:
#         descriptions = data['Description'].tolist()
#         amounts = data['Amount'].tolist()

#         # Fit and transform the scaler with the current amounts
#         all_amounts = [amt for amt in amounts]
        
#         # Get predictions for the entire dataset
#         predictions = [call_gemini_api(desc, amt, all_amounts) for desc, amt in zip(descriptions, amounts)]

#         # Debugging output for predictions
#         st.write("Predictions:")
#         st.write(predictions)

#         # Handle unexpected responses
#         categories = []
#         subcategories = []
#         for pred in predictions:
#             try:
#                 category, subcategory = pred.split(',')
#                 categories.append(category.strip())
#                 subcategories.append(subcategory.strip())
#             except ValueError:
#                 categories.append("Unknown")
#                 subcategories.append("Unknown")

#         data['Predicted Category'] = categories
#         data['Predicted Subcategory'] = subcategories

#         st.write("Data with Predictions:")
#         st.write(data)
        
#         # Update and display balance sheet
#         balance_sheet = update_balance_sheet(data)

#         # Formatting the balance sheet
#         st.write("Balance Sheet:")
#         for category in balance_sheet['Predicted Category'].unique():
#             st.write(f"### {category}")
#             category_data = balance_sheet[balance_sheet['Predicted Category'] == category]
#             for _, row in category_data.iterrows():
#                 st.write(f"{row['Predicted Subcategory']}: ${row['Amount']:,.2f}")
#             total = category_data['Amount'].sum()
#             st.write(f"**Total {category}: ${total:,.2f}**")

# # User input prompt
# description_input = st.text_input("Enter a transaction description")
# amount_input = st.number_input("Enter the transaction amount", min_value=0.0, format="%.2f")

# if st.button("Predict Category and Subcategory"):
#     if description_input and amount_input:
#         # Fit and transform with a single amount
#         all_amounts = [amount_input]
#         pred = call_gemini_api(description_input, amount_input, all_amounts)
#         try:
#             category, subcategory = pred.split(',')
#             st.write(f"Predicted Category: {category.strip()}")
#             st.write(f"Predicted Subcategory: {subcategory.strip()}")
#         except ValueError:
#             st.write("Prediction error: Unable to parse the response")

# if __name__ == '__main__':
#     st_main()
