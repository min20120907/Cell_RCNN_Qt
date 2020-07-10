import json
import glob

result = {}

for f in glob.glob("*.json"):
    with open(f, "r") as infile:
        result.update(json.load(infile))
        

with open("merged_file.json", "w") as outfile:
     json.dump(result, outfile)
