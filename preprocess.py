import re
from pathlib import Path

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z \n]', '', text) # leave only lowercase letters and whitespace
    text = re.sub(r'\s+', ' ', text) # convert multiple whitespace characters to a single space

    return text.strip()

def prepare_data(input_folder, output_folder):

    path_in = Path(input_folder)
    path_out = Path(output_folder)
    
    path_out.mkdir(parents=True, exist_ok=True)
    text_files = list(path_in.glob('*.txt'))
    
    if not text_files:
        print(f"Have not found any text files in folder {input_folder}")
        return

    print(f"Found {len(text_files)} text files.")

    for file in text_files:
        print(f"Running: {file.name}")
        
        with open(file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        cleaned_text = clean_text(raw_text)
        
        new_path = path_out / file.name
        
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            
    print(f"Finished! Cleaned files are in folder: {output_folder}")

if __name__ == '__main__':
    prepare_data('raw_data', 'cleaned_data')