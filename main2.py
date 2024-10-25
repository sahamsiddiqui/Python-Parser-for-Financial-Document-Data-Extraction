# loading Libraries 
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Same regex and unwanted substrings definitions
eps_regex = r'(?:net|core|comprehensive)?\s*(earnings?|income|profit|loss|revenue|basic|diluted|adjusted|gaap)([\s\S]*?)(earnings?|income|profit|loss|revenue|basic|diluted|adjusted|gaap)?\s*(.*)?(?:\s*per\s*(.*?)\s*share)?'
unwanted_substrings = ['shares used to compute', 'shares used to calculate', 'book value']

# Function to extract EPS value from a <td> cell
def get_numeric_value_from_row(row):
    eps_value_regex = r'[-(]?\$?\d+\.\d{2}[)]?'
    for td in row.find_all('td'):
        td_text = td.get_text(strip=True)
        match = re.search(eps_value_regex, td_text)
        if match:
            eps_value = match.group(0)
            if eps_value.startswith('(') and eps_value.endswith(')'):
                eps_value = '-' + eps_value[1:-1]
            eps_value = eps_value.replace('$', '')
            return eps_value
    return None

# Function to process a single HTML file
def process_file(filepath, df_eps):
    file_has_data = False  # Flag to track if the file contains EPS data

    # Parse the HTML file using lxml parser
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')  # Using 'lxml' parser for faster performance

    tables = soup.find_all('table')

    for table in tables:
        rows = table.find_all('tr')
        for i, row in enumerate(rows):
            row_text = ' '.join([cell.get_text(strip=True) for cell in row.find_all('td')])

            if re.search(eps_regex, row_text, re.IGNORECASE):
                if any(re.search(substring, row_text, re.IGNORECASE) for substring in unwanted_substrings):
                    continue

                eps_value = get_numeric_value_from_row(row)

                if eps_value is None:
                    if i + 1 < len(rows):
                        next_row = rows[i + 1]
                        next_row_text = ' '.join([td.get_text(strip=True) for td in next_row.find_all('td')])
                        eps_value = get_numeric_value_from_row(next_row)
                        row_text = f"{row_text} {next_row_text}"

                if eps_value:
                    df_eps.loc[len(df_eps)] = [os.path.basename(filepath), row_text, eps_value]
                    file_has_data = True

    if not file_has_data:
        df_eps.loc[len(df_eps)] = [os.path.basename(filepath), np.nan, np.nan]

# Main function to iterate through files in the 'data' directory
def process_all_files():
    data_dir = './data'   #provided folder name 
    if not os.path.exists(data_dir):
        print(f"The directory '{data_dir}' does not exist.")
        return

    df_eps = pd.DataFrame(columns=['filename', 'potential_eps_text', 'potential_eps_value'])
    for filename in os.listdir(data_dir):
        if filename.endswith('.html'):
            filepath = os.path.join(data_dir, filename)
            process_file(filepath, df_eps)
    return df_eps

# Function to clean potential_eps_text
def clean_eps_text(text):
    # Trim to first 55 characters
    trimmed_text = text[:55]
    
    # Replace all digits with nothing
    cleaned_text = re.sub(r'\d+', '', trimmed_text)
    
    return cleaned_text

# Cosine similarity calculation functions
true_eps_list = [
"(LOSS) EARNINGS PER SHARE basic",
"Basic Earnings per Share",
"Basic and diluted net income per share",
"Net income per share—basic",
"Basic and diluted net loss percommon share",
"Basic earnings (loss) per share",
"Basic earnings per share",
"Earnings per share - basic",
"Net income (loss) per share - basic",
"Basic and diluted earnings per share",
"Net loss per share: Basic",
"Basic net income per share",
"Net (loss) income per share",
"NET EARNINGS PER COMMON SHARE - BASIC",
"Loss Per Common Share: Basic",
"Basic income (loss) per share",
"Net income per common share: Basic",
"Net income available to common stockholders per share",
"Basic earnings per share",
"Earnings (loss) per common share Basic",
"GAAP loss per share — basic and diluted",
"Net loss per common share: Basic",
"Basic net income per share attributable to",
"Income (loss) per share—basic",
"Basic Earnings per Share",
"Earnings per share: Basic",
"Earnings per common share",
"Earnings per share: Basic",
"Basic and diluted net (loss) income per share",
"basic and diluted net (loss) income per common share",
"Basic and diluted net loss per share",
"Net income (loss) per common share - basic",
"Net income per common share:(1) Basic",
"Earnings per share: Basic",
"Basic earnings per share:",
"Net income allocated to shareholders per share: Basic",
"Basic and diluted loss per share:",
"Earnings per ordinary share",
"Net loss per common shareholders",
"GAAP earning per share",
"GAAP income (loss) per share",
"Net income allocated to shareholders per share: Basic"
    
]


dissimilar_words = ["tax", "per unit", "ARPU", "assets", "Unrealized", "paging", "loan", "interest", "derivative", "allowance"]

secondary_eps_list = ["dilute", "non-GAAP", "comprehensive", "adjusted"]

# Vectorize the True EPS List (reference texts) only
vectorizer = TfidfVectorizer()
true_eps_tfidf = vectorizer.fit_transform(true_eps_list)

def calculate_similarity(potential_text, true_eps_tfidf, vectorizer):
    potential_text_tfidf = vectorizer.transform([potential_text])
    similarities = cosine_similarity(potential_text_tfidf, true_eps_tfidf)
    return np.max(similarities)

def apply_word_penalty(text, dissimilar_words, secondary_eps_list):
    penalty = 0
    text_lower = text.lower()
    
    for word in dissimilar_words:
        if word.lower() in text_lower:
            penalty += 0.2
    
    for word in secondary_eps_list:
        if word.lower() in text_lower:
            penalty += 0.1

    return penalty

# Function to clean and transform EPS values
def clean_eps_value(row):
    potential_eps_value = row['potential_eps_value']
    potential_eps_text = row['potential_eps_text'].lower()
    
    if pd.isna(potential_eps_value):
        return potential_eps_value
    
    eps_value = potential_eps_value.strip()
    
    if eps_value.startswith('('):
        eps_value = eps_value.replace('(', '-')
    
    try:
        numeric_value = float(eps_value)
    except ValueError:
        return potential_eps_value
    
    if 'loss' in potential_eps_text and 'gain' not in potential_eps_text and 'profit' not in potential_eps_text:
        if numeric_value > 0:
            numeric_value = -numeric_value
    
    return numeric_value

def main():
    df_eps = process_all_files()
    
    # Clean potential_eps_text
    df_eps['potential_eps_text'] = df_eps['potential_eps_text'].apply(clean_eps_text)
    
    # Calculate similarity scores
    similarity_scores = []
    for i, row in df_eps.iterrows():
        potential_text = row['potential_eps_text']
        max_similarity = calculate_similarity(potential_text, true_eps_tfidf, vectorizer)
        penalty = apply_word_penalty(potential_text, dissimilar_words, secondary_eps_list)
        final_score = max_similarity - penalty
        similarity_scores.append(max(0, final_score))

    df_eps['similarity_score'] = similarity_scores
    
    # Clean EPS values
    df_eps['cleaned_eps_value'] = df_eps.apply(clean_eps_value, axis=1)

    # Keep the row with the highest similarity score per file
    final_df_eps = df_eps.loc[df_eps.groupby('filename')['similarity_score'].idxmax()].reset_index(drop=True)
    
    # Save to CSV
    final_df_eps[['filename', 'cleaned_eps_value']].rename(columns={'cleaned_eps_value': 'EPS'}).to_csv('output_csv.csv', index=False)

    print(" The Final CSV has been saved !! ")
if __name__ == "__main__":
    main()
