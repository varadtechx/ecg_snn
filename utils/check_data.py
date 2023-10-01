import os

folder_path = "/home/ubuntu/Desktop/Projects/SNN/archive/mitbih_database/mitbih_database/"


csv_count = 0
txt_count = 0


for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        csv_count += 1
    elif filename.endswith(".txt"):
        txt_count += 1

print(f"Number of CSV files: {csv_count}")
print(f"Number of TXT files: {txt_count}")
