import os

sentences = {}
path = {}
path['root'] = os.getcwd()
for tpart in ['train', 'test']:
  path[tpart] = os.path.join(path['root'], tpart)
  sentences[tpart] = []
  folderList = os.listdir(path[tpart])
  for folder in folderList:
    fileList = os.listdir(os.path.join(path[tpart], folder))
    for eachf in fileList:
      fpath = os.path.join(path[tpart], folder, eachf)
      with open(fpath, 'r') as f:
        sentences[tpart].append(f.read())
  #save sentences in file
  sentencePath = os.path.join(path['root'], 'sentences-{}'.format(tpart))
  with open(sentencePath, 'w') as f:
    for sentence in sentences[tpart]:
      f.write(sentence)
      f.write('\n')

