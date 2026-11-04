import json, sys

a = json.load(open(sys.argv[1]))["results"]
b = json.load(open(sys.argv[2]))["results"]
for x, y in zip(a, b):
    if x["answerable"] and x["rank"] != y["rank"]:
        print(f"{x['rank']} -> {y['rank']}   {x['question']}")