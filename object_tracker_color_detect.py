from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import math
from math import log10, floor

import operator

class Colors(object):
	class Color(object):
		def __init__(self, value):
			self.value = value

		def __str__(self):
			return "%s : %s" % (self.__class__.__name__, self.value)

	class Red(Color): pass
	class Blue(Color): pass
	class Green(Color): pass
	class Yellow(Color): pass
	class White(Color): pass
	class Gray(Color): pass
	class Black(Color): pass
	class Pink(Color): pass
	class Teal(Color): pass

class ColorWheel(object):
	def __init__(self, rgb):
		r, g, b = rgb

		self.rgb = (Colors.Red(r), Colors.Green(g), Colors.Blue(b), )
	
	def estimate_color(self):
		dominant_colors = self.get_dominant_colors()

		total_colors = len(dominant_colors)
		
		if total_colors == 1:
			return dominant_colors[0]
		elif total_colors == 2:
			color_classes = [x.__class__ for x in dominant_colors]

			if Colors.Red in color_classes and Colors.Green in color_classes:
				return Colors.Yellow(dominant_colors[0].value)
			elif Colors.Red in color_classes and Colors.Blue in color_classes:
				return Colors.Pink(dominant_colors[0].value)
			elif Colors.Blue in color_classes and Colors.Green in color_classes:
				return Colors.Teal(dominant_colors[0].value)
		elif total_colors == 3:
			if dominant_colors[0].value > 200:
				return Colors.White(dominant_colors[0].value)
			elif dominant_colors[0].value > 100:
				return Colors.Gray(dominant_colors[0].value)
			else:
				return Colors.Black(dominant_colors[0].value)
		else:
			print("Dominant Colors : %s" % dominant_colors)
	
	def get_dominant_colors(self):
		max_color = max([x.value for x in self.rgb])

		return [x for x in self.rgb if x.value >= max_color * .9]

def round_to_1(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def process_image(image):
	image_color_quantities = {}

	width, height = image.size

	# for x in range(width):
		# for y in range(height):

	width_margin = 1
	height_margin = 1
	# print height
	# print range(height_margin, height - height_margin)
	for x in range(width_margin, width - width_margin):
		for y in range(height_margin, height - height_margin):
			r, g, b = image.getpixel((x, y))

			key = "%s:%s:%s" % (r, g, b, )

			key = (r, g, b, )

			image_color_quantities[key] = image_color_quantities.get(key, 0) + 1

	total_assessed_pixels = sum([v for k, v in image_color_quantities.items()])

	# strongest_color_wheels = [(ColorWheel(k), v / float(total_pixels) * 100, ) for k, v in test.items() if v > 30]
	strongest_color_wheels = [(ColorWheel(k), v / float(total_assessed_pixels) * 100, ) for k, v in image_color_quantities.items()]

	final_colors = {}

	for color_wheel, strength in strongest_color_wheels:
		# print "%s => %s" % (strength, [str(x) for x in color_wheel.get_dominant_colors()], )

		# print "%s => %s" % (strength, color_wheel.estimate_color(), )

		color = color_wheel.estimate_color()

		final_colors[color.__class__] = final_colors.get(color.__class__, 0) + strength

	#for color, strength in final_colors.items():
		#print("%s - %s" % (color.__name__, strength, ))

	#image.show()
	max_color=max(final_colors.items(), key=operator.itemgetter(1))[0]

	return (max_color.__name__, final_colors[max_color])

prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

# load weights and set defaults
config_path='config/yolov3-spp.cfg'
weights_path='config/yolov3-spp.weights'
class_path='config/coco.names'
img_size=608
conf_thres=0.1
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
#model.cuda()
model.eval()

classes = utils.load_classes(class_path)
#Tensor = torch.cuda.FloatTensor
Tensor = torch.FloatTensor

def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

videopath = 'onbridge.mp4'

import cv2
from sort import *
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

frames = 0
starttime = time.time()
counter=0
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            counter = counter + 1
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]

            margin=0
            small_pic = frame[y1+margin:y1+box_h-2*margin, x1+margin:x1+box_w-2*margin]
            #cv2.imwrite("zzz"+str(counter)+".jpg", small_pic)
            cv2_im = cv2.cvtColor(small_pic,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            answer = process_image(pil_im)

            #color_histogram_feature_extraction.color_histogram_of_test_image(small_pic)
            #prediction = knn_classifier.main('training.data', 'test.data')
            #cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+80, y1), color, -1)
            #cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), (0, 255, 0), 4)
            cv2.putText(frame, '[' + str(answer[0]) + ", " + str(round_to_1(answer[1])) + '%]', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break

totaltime = time.time()-starttime
print(frames, "frames", totaltime/frames, "s/frame")
cv2.destroyAllWindows()
outvideo.release()
