import os
import shutil

import pandas as pd
from tqdm import tqdm


def clean_paths(df, path_column='path'):
    df[path_column] = df[path_column].str.replace(r'^\.\./', '', regex=True)
    return df


def copy_images(df, target_dir, path_column='path', name_column='image_name'):
    os.makedirs(target_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Copying images'):
        src = row[path_column]
        dst = os.path.join(target_dir, row[name_column] + '.jpg')
        if not os.path.exists(dst):
            try:
                shutil.copyfile(src, dst)
            except Exception as e:
                print(f"[ERROR] Failed to copy {src} -> {dst}: {e}")


def main():
    # Change for spesific dataframe (mel, nev, binary, benign, malignant, scc, all, etc.)
    df_name = "##"

    df = pd.read_csv(
        f'data/{df_name}_classification/{df_name}_classification.csv')
    df = clean_paths(df)
    copy_images(
        df, target_dir=f'data/{df_name}_classification/{df_name}_images/')


if __name__ == '__main__':
    main()
