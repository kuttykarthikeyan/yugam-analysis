import pandas as pd
import spacy

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Read the CSV and Excel files
workshop_df = pd.read_csv('yugam_title.csv')
response_df = pd.read_excel('tokenized_and_lemmatized_data.xlsx')

# Initialize an empty DataFrame to store the results
output_data = []

# Iterate through each row in workshop titles
for workshop_title in workshop_df['workshop-titles']:
    # Ensure workshop title is a string
    if isinstance(workshop_title, str):
        # Process the workshop title using spaCy
        doc_title = nlp(workshop_title)
        
        # Initialize a list to store related words
        related_words = []
        
        # Iterate through each row in tokenized responses
        for response in response_df['Tokenized Response']:
            # Ensure response is a string
            if isinstance(response, str):
                # Process the response using spaCy
                doc_response = nlp(response)
                
                # Calculate the number of tokens to include (50% of the original content)
                num_tokens_to_include = len(doc_response) // 2
                
                # Extract words from the response that are relevant to the workshop title
                response_words = [token.text for token in doc_response if token.text in workshop_title][:num_tokens_to_include]
                
                # Add the relevant words to the list of related words
                related_words.extend(response_words)
        
        # Remove duplicate words
        related_words = list(set(related_words))
        
        # Append the result for this title to the output data
        output_data.append({'Workshop Title': workshop_title, 'Related Words': related_words})
    else:
        output_data.append({'Workshop Title': '', 'Related Words': []})

# Create DataFrame for output
output_df = pd.DataFrame(output_data)

# Save the output to an Excel file
output_df.to_excel('workshop_related_words_with_limited_context.xlsx', index=False)
