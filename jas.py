import json

# Set the correct file path
file_path = r"C:\Users\Sarath S\Downloads\extracted_text_paddle.txt"

# Read the extracted text file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read().split("[Extracted from")  # Split by each article
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Check the path and try again.")
    exit()

# Process the text into structured format
articles = []
for entry in data[1:]:  # Skip the first split since it's empty before the first "[Extracted from"
    lines = entry.split("\n", 1)
    image_name = lines[0].strip(" ]") if len(lines) > 1 else "unknown"
    content = lines[1].strip() if len(lines) > 1 else ""
    if content.strip():  # Avoid empty texts
        articles.append({"image": image_name, "text": content})

# Save as JSON
json_path = "articles.json"
with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(articles, json_file, indent=4, ensure_ascii=False)

print(f"✅ Extracted text saved in '{json_path}'")
