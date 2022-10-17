from music21 import converter
import warnings
warnings.filterwarnings('ignore')

file = converter.parse(r'C:\Users\samue\Documents\Applied Data Science\INFO-H518 Deep Learning\Project\xy.mid')
components = []
for element in file.recurse():
    components.append(element)
    print(element)