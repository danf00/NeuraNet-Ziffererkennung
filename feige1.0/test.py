import os 
from pathlib import Path

path = Path(__file__).parent
path = path.joinpath("variants")
variants = os.listdir(path)
buffer = []
for item in variants:
    item = item.replace("k", "")
    item = int(item)
    buffer.append(item)

buffer.sort()
print(buffer)

y = []

for x in buffer:
    x = str(x) + "k"
    y.append(x)

print(y)

