from deepface import DeepFace
import json

img_path_1 = 'images/anand.jpg'
img_path_2 = './images'

# Face matching
# result = DeepFace.verify(img_path_1, img_path_2)
# print(json.dumps(result, indent=2))

# find face in db

# dfs = DeepFace.find(img_path_1, img_path_2)
# print(dfs)

# face analysis


objs = DeepFace.analyze(img_path_1,)
print(json.dumps(objs, indent=2))