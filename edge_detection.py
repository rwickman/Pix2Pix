import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import cv2 
import tensorflow as tf
import glob
from tqdm import tqdm
import argparse

#data_root = 'naruto_dataset'
data_root = 'fake'
image_paths = [os.path.join(data_root, f) for f in os.listdir(data_root)]
#image_paths = glob.glob("naruto_dataset/*/*.jpg") + glob.glob("naruto_dataset/*/*.png")#os.listdir(data_root)
images = []
img_width = 256
img_height = 256
output_dir = "output_edges"

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#! [CropLayer]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

class EdgeDetection:
    def __init__(self, args):
        self._args = args
        if not self._args.use_canny:
            cv2.dnn_registerLayer('Crop', CropLayer)
            # Load caffe model
            self._net = cv2.dnn.readNetFromCaffe(self._args.model_def, self._args.pretrained_model)  
        
        self._get_img_paths()
    
    def _get_img_paths(self):
        # Get JPEG images
        self._img_paths = glob.glob(os.path.join(output_dir, "**/*.jpg"), recursive=True) 
        
        # Get PNGs
        self._img_paths = glob.glob(os.path.join(output_dir, "**/*.png"), recursive=True) 

    def create_edges(self):
        with tf.io.TFRecordWriter(os.path.join(self._args.output_dir, self._args.output_tfrecords)) as writer:
            for image_path in tqdm(image_paths):
                image_name = image_path.split("/")[-1]
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                if not self._args.use_canny:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    inp = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(img_width, img_height),
                                            mean=(104.00698793, 116.66876762, 122.67891434),
                                            swapRB=False, crop=False)
                
                    # Predict edges    
                    self._net.setInput(inp)
            
                    out = self._net.forward()
                    out = out[0, 0]
                    out = (out * 255).astype(np.uint8)
                else:
                    out = cv2.Canny(image, self._args.img_dim, self._args.img_dim)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #print(out.shape)


                #out = cv2.resize(out, (image.shape[1], image.shape[0]))
                image_split = image_name.split(".")
                #image_name = "".join(image_split[:-1]) + "_edges." + image_split[-1]
                image_name = "".join(image_split[:-1]) + "_edges"
                
                
                #images.append(image)
                # Convert to RGB
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

                # print(image.shape)
                # print(out.shape)
                
                img_feature = {
                    "height": _int64_feature(image.shape[1]),
                    "width": _int64_feature(image.shape[0]),
                    "real_image" : _bytes_feature(image.tostring()),
                    "input_image" : _bytes_feature(out.tostring())
                }
            
                
                #example = tf.train.Example(features=tf.train.Features(feature=img_feature))
                #writer.write(example.SerializeToString())
                
                #np.save(os.path.join(output_dir, image_name), img_pair)
                #cv2.imwrite(os.path.join(output_dir, image_name), img_pair)
                #cv2.imshow('Holistically-Nested Edge Detection', out)
                if self._args.plot_edges:
                    fig, axs = plt.subplots(1,2)
                    axs[0].imshow(image/255)
                    axs[1].imshow(out/255, cmap='gray')
                    plt.show()
                
                #out = cv2.resize(out, 
                
                #images.append(image)

def main(args):
    output_path = os.path.join(args.output_dir, args.output_tfrecords)
    if os.path.exists(output_path):
        raise Exception("Error {} already exists!".format(output_path))
    edge_detector = EdgeDetection(args)
    edge_detector.create_edges()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="naruto_dataset",
        help="Input images directory.")
    parser.add_argument("--output_dir", default="output_edges/train",
        help="Output TFRecords directory.")
    parser.add_argument("--output_tfrecords", default="edges.tfrecords",
        help="Output TFRecords name.")
    parser.add_argument("--use_canny", action="store_true",
        help="Use Canny edge detection instead of HED.")
    parser.add_argument("--img_dim", type=int, default=256,
        help="Output edges width/height.")
    parser.add_argument("--model_def", default="models/edge_detection/deploy.prototxt",
        help="Protobuf file with model definition.")
    parser.add_argument("--pretrained_model", default="models/edge_detection/hed_pretrained_bsds.caffemodel",
        help="Pretrained Caffee HED model.")
    parser.add_argument("--plot_edges", action="store_true",
        help="Plot image and edges.")
    main(parser.parse_args())