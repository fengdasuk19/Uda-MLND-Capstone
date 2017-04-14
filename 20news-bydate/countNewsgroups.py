import os

path = {}
path['train'] = os.path.join(os.getcwd(), '20news-bydate-train')
path['test'] = os.path.join(os.getcwd(), '20news-bydate-test')

for tpart in path:
  with open('report-{}.txt'.format(tpart), 'w') as report:
    report.write('This is a report for the part: {}\n\n'.format(tpart))
    mdTableHead = 'Group | News count\n-----|-----\n'
    report.write(mdTableHead)
    for groupName in os.listdir(path[tpart]):
      newsCount = len(os.listdir(os.path.join(path[tpart], groupName)))
      report.write('{} | {}\n'.format(groupName, newsCount))
    report.write('\n')
