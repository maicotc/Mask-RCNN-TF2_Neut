import os
import sys
import json
import time
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN

#from mrcnn import model as modellib, utils


class CustomDataset(Dataset):

    def load_custom(self, dataset_dir):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the number of classes required to detect
        self.add_class("custom", 1, "N1")
        self.add_class("custom", 2, "N2")
        
        N1_dir = dataset_dir + '/N1/images/'
        N1_annotations_dir = dataset_dir + '/N1/annots/'
        
        N2_dir = dataset_dir + '/N2/images/'
        N2_annotations_dir = dataset_dir + '/N2/annots/'
        
        self.alocate_image(N1_dir, N1_annotations_dir)
        self.alocate_image(N2_dir, N2_annotations_dir)
        
    def alocate_image(self, image_directory, annotation_directory):
        for filename in os.listdir(image_directory):
            image_id = filename[:-4]
            
            img_path = image_directory + filename
            ann_path = annotation_directory + image_id + '.json'

            self.add_image('custom', image_id=image_id, path=img_path, annotation=ann_path)
      

    def load_mask(self, image_id):
        class_ids = []
        
        info = self.image_info[image_id]
        #print(f" ************** AQUI {info}")
        boxes, w, h, labels = self.extract_boxes(info["annotation"])
        #print(f" ************** AQUI {boxes}")
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = int(box[1]), int(box[3])
            col_s, col_e = int(box[0]), int(box[2])
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(labels[i]))
        return masks, np.asarray(class_ids, dtype='int32')

    
    def extract_boxes(self, filename):
        boxes_list = []
        label_list = []
        
        #json file need path
        file = open(filename)
        annotation = json.load(file)
        
        for annot in annotation['shapes']:
            label = annot['label']
            xmin = annot['points'][0][0]
            ymin = annot['points'][0][1]
            xmax = annot ['points'][1][0]
            ymax = annot['points'][1][1]
            coords = [xmin, ymin, xmax, ymax]
            boxes_list.append(coords)
            label_list.append(label)
        height = annotation['imageHeight']
        width = annotation['imageWidth']
        return boxes_list, width, height, label_list
   

class NeutConfig(Config):
    NAME = "neut_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2 + 1

    STEPS_PER_EPOCH = 64
    
    LEARNING_RATE = 2e-5

#Training
train_dataset = CustomDataset()
train_dataset.load_custom(dataset_dir='Dataset/train')
train_dataset.prepare()

# Validation
validation_dataset = CustomDataset()
validation_dataset.load_custom(dataset_dir='Dataset/validation')
validation_dataset.prepare()

neut_config = NeutConfig()

model = MaskRCNN(
    mode='training',
    model_dir='./',
    config=neut_config
    )

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])            

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=neut_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

NAME = f"Maskrcnn_neut-{int(time.time())}"

model_path = "{}".format(NAME)
model.keras_model.save_weights(model_path)           
