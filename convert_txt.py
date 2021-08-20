import os
from os import walk, getcwd
import shutil
from PIL import Image

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
    
"""-------------------------------------------------------------------""" 

""" Configure Paths"""   
mypath = "./dataset/"
outpath = "./result/"
json_backup ="./json_backup/"

wd = getcwd()
#list_file = open('%s_list.txt'%(wd), 'w')

""" Get input json file list """
json_name_list = []
for file in os.listdir(mypath):
    if file.endswith(".json"):
        json_name_list.append(file)
    

""" Process """
for json_name in json_name_list:
    txt_name = json_name.rstrip(".json") + ".txt"
    """ Open input text files """
    txt_path = mypath + json_name
    #print("Input:" + txt_path)
    txt_file = open(txt_path, "r")
    
    """ Open output text files """
    txt_outpath = outpath + txt_name
    #print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "w+")

    """ Convert the data to YOLO format """ 
    lines = txt_file.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
    for idx, line in enumerate(lines):
        if ("lineColor" in line):
            break 	#skip reading after find lineColor
        if ("label" in line):
            x1 = float(lines[idx+3].rstrip(','))
            y1 = float(lines[idx+4])
            #x2 = float(lines[idx+9].rstrip(','))
            #y2 = float(lines[idx+10])

            label = line.split('"')

            cls = label[3]

	    #in case when labelling, points are not in the right order

            img_path = str('%s/dataset/%s.jpg'%(wd, os.path.splitext(json_name)[0]))

            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])

            nx = x1/w
            ny = y1/h
            #print(w, h)
            #print(nx, ny)
            
            bb = (nx,ny)

            txt_outfile.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')
    txt_file.close()
    txt_outfile.close()

    txt_outfile = open(txt_outpath, 'r+')
    xtlines = txt_outfile.readlines()
    res = []
    n = 1

    for line in xtlines:
        txtsplit = line.split(" ")
        #print(n,txtsplit[0])
        if(str(n)!=txtsplit[0] and n <= len(xtlines)):
            #print("T")
            res.append(str(n)+" "+ "0"+" "+ "0"+"\n")
            n += 1
        res.append(line)
        n += 1
    for xn in range(n,16):
        res.append(str(xn)+" "+ "0"+" "+ "0")
        if(xn<15):
            res.append("\n")
    #print(res)
    txt_outfile.close()
    txt_outfile = open(txt_outpath,'w')
    # txt_outfile.truncate(0)
    for item in res:
        txt_outfile.write(str(item))
    txt_outfile.close()

    #os.rename(txt_path,json_backup+json_name)	#move json file to backup folder

label_path = "./label.txt"

label_file = open(label_path, "w+") 
label = []

for file_path in os.listdir(outpath):
    context = ""
    txt_file = open("./result/"+file_path, "r")
    xtlines = txt_file.readlines()
    file_path = file_path.split(".")
    context = context + file_path[0]
    for line in xtlines:
        line_split = line.split("\n")
        line_split = line_split[0].split(" ")
        #print(len(line_split))
        context = context + " " + line_split[1] + " " + line_split[2]
        #print(context)
    label.append(context)
    txt_file.close()

#print(len(label))
line_count = 0

for item in label:
    split = item.split(" ")
    print(split[0], len(split))
    label_file.write(str(item))
    label_file.write("\n")
    line_count = line_count + 1

label_file.close()

#shutil.rmtree('./result')
#os.mkdir('./result')

print("finish convert with", line_count, "images")