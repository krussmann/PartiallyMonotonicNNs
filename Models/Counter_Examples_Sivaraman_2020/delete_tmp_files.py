import os

# Directory where the files are located
directory_path = "C:\\Users\\russm\\AppData\\Local\\Temp"

crit = "b'(set-option :produce-models true)\\r\\n'"

# Function to determine if file should be deleted
def is_delete(file_path):
    decision = False
    try:
        with open(file_path, "rb") as f:
            lines = f.readlines()
            if str(lines[0]) == crit:
                decision = True
    except:
        print("Permission denied")
    return decision

# Iterate through files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    print(filename)
    if filename[:3] == 'tmp':
        me = None

    # Check if it's a file (not a subdirectory)
    if os.path.isfile(file_path):
        delete = False
        delete = is_delete(file_path)

        # Check if the hash already exists in the dictionary
        if delete:
            # If a file with the same hash already exists, delete the current file
            os.remove(file_path)
            print(f"Deleted file: {filename}")

print("Files have been deleted.")
