import os
import sys
from collections import defaultdict
import pprint


#USAGE: arg1: path to subfolders containing the annotation files


def get_files(path):

    #check if path exists
    assert  os.path.exists('./'+ path), "first argument has to be a path"

    annfiles=[]
    txtfiles=[]

    for f in os.listdir('./'+ path):
        if '.txt' in f:
            txtfiles.append(f)
        elif '.ann' in f:
            annfiles.append(f)

    #sanity check
    assert len(annfiles) == len(txtfiles), "not the same number of text({}) and annotation files({})".format(len(txtfiles), len(annfiles))

    return txtfiles, annfiles

def read_annotations(annfile):
    #print('Reading '+ annfile + '...')
    components = defaultdict(int)
    relations = defaultdict(int)

    with open(annfile, 'r') as f:
        annotations = f.readlines()

        for anno in annotations:
            label = anno.split('\t')[1].split()
            if 'Arg' in label[1]:
                relations[label[0]] += 1
            else:
                components[label[0]] += 1

    return components, relations

def get_total_count(annotations):
    total = defaultdict(int)

    for doc in annotations.keys():
        #componentsclear
        for ann in annotations[doc][0].keys():
            total[ann] += annotations[doc][0][ann]
        #relations
        for ann in annotations[doc][1].keys():
            total[ann] += annotations[doc][1][ann]

    return total

def main(path):
    print(path)
    txt, anns = get_files(path)

    annotations = dict()

    for annfile in anns:
        comp, rel = read_annotations(path + annfile)
        annotations[annfile[:-4]] = comp, rel

    total = get_total_count(annotations)
    pprint.pprint(dict(total))

path = './' + sys.argv[1]

subdirs = [name for name in os.listdir(path) if os.path.isdir(path+name)]
for dir in subdirs:
    main(path + dir + '/')
