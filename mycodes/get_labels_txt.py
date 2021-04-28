import os

f = open("../labels.txt",'w')
alllabels = os.listdir("../data_prepare/raptile_images")
alllabels.sort()
for label in alllabels:
    f.write(label + '\n')
f.close()