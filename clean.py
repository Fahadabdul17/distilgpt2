import pandas as pd
import re

def load_data(file_path):
    # Membaca file CSV dengan delimiter pipa dan menangani baris buruk
    return pd.read_csv(file_path, header=None, delimiter='|', on_bad_lines='skip')

def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.strip()
    
    # Hapus karakter yang tidak diinginkan dan format khusus
    text = re.sub(r'(package|import|public class|private|protected|public|extends|implements|return|void|static|final|throws|super|try|catch|finally|throw|synchronized|volatile|native|abstract|interface|enum|this|new|if|else|for|while|do|switch|case|break|continue|default|try|catch|finally|throw|throws|public|private|protected|package|import|interface|enum|class|extends|implements|super|this|new|if|else|switch|case|default|for|while|do|break|continue|return|void|abstract|final|native|synchronized|transient|volatile|assert|strictfp|throw|throws|super|interface|enum|class|extends|implements|new|super|try|catch|finally|throw|throws|return)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'//.*|/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'\n \n', '', text)  # Hapus pola /nn
    text = re.sub(r'[^\w\s]', '', text)  # Hapus karakter non-alphanumeric kecuali whitespace
    text = re.sub(r'\s+', ' ', text)  # Hapus whitespace berlebih
    
    return text.strip()

def clean_dataset(df):
    # Terapkan pembersihan ke setiap elemen DataFrame
    df_cleaned = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
    
    # Hapus baris yang berisi angka
    df_cleaned = df_cleaned[~df_cleaned.apply(lambda row: row.str.contains(r'\b\d+\b').any(), axis=1)]
    
    # Hapus kolom yang sepenuhnya kosong
    df_cleaned = df_cleaned.dropna(axis=1, how='all')
    
    return df_cleaned

def save_cleaned_data(df, output_file_path):
    # Simpan file CSV dengan delimiter pipa
    df.to_csv(output_file_path, index=False, header=False, sep='|')

def main():
    input_file = 'dataset-a.csv'  # Ganti dengan jalur file yang sesuai
    output_file = 'cleaned_dataset.csv'  # Ganti dengan jalur file yang sesuai
    
    df = load_data(input_file)
    df_cleaned = clean_dataset(df)
    save_cleaned_data(df_cleaned, output_file)
    print(f"Dataset telah dibersihkan dan disimpan di {output_file}")

if __name__ == "__main__":
    main()
