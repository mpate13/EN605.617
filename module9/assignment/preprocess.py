import pandas as pd
import numpy as np

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    # Convert string-based lists "[...]" into actual Python lists
    df['genres'] = df['genres'].apply(
        lambda x: eval(x) if isinstance(x, str) else []
    )
    
    # Extract unique genres and create the index map
    all_genres = sorted(list(set(
        [g for sublist in df['genres'] for g in sublist]
    )))
    genre_map = {genre: i for i, genre in enumerate(all_genres)}

    # Save the ID map so you can look up indices for user input
    with open('genre_map.txt', 'w') as f:
        for g, i in genre_map.items():
            f.write(f"{i}: {g}\n")

    # Build CSR Matrix components
    row_ptr, col_ind, values = [0], [], []
    for _, row in df.iterrows():
        for g in row['genres']:
            col_ind.append(genre_map[g])
            values.append(1.0)
        row_ptr.append(len(col_ind))

    # Export binaries for CUDA
    np.array(row_ptr, dtype=np.int32).tofile('row_ptr.bin')
    np.array(col_ind, dtype=np.int32).tofile('col_ind.bin')
    np.array(values, dtype=np.float32).tofile('values.bin')
    df['title'].to_csv('titles.txt', index=False, header=False)
    print(f"Success! {len(all_genres)} genres mapped in genre_map.txt")

if __name__ == "__main__":
    prepare_data('books_1.Best_Books_Ever.csv')