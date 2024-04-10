# Required Packages
import argparse
import pathlib
import pandas as pd
import numpy as np
import PIL.Image
import tensorflow
import glob, time, os, sys, getopt
import tensorflow as tf
import requests
import json
import logging
import cv2

from os.path import exists
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from deepsort_tracker import DeepSort


# Package for securing function key
##from dotenv import dotenv_values

# Global Variables
model_file_path = None
nvr_path = None
output_images_path = None
car_counter_variables_path = None
probability_threshold = None
zone_number = None
parking_lot_type = None
traffic_flow_direction = None
logger = None
log_path = None
log_level = None
skip_image_counter = None
# number indicating which GPU to use
gpu_number = None
# amount of gpu memory to use. specified in MB. default is 512MB
gpu_mem_limit = 1024

# base URL for API to store (num_cars_in, num_cars_out)
base_url = None

# car counting line coordinates
counting_line_coordination_x1 = None
counting_line_coordination_y1 = None
counting_line_coordination_x2 = None
counting_line_coordination_y2 = None

# traffic lane separation line coordinates
lane_separation_line_coordination_x1 = None
lane_separation_line_coordination_y1 = None
lane_separation_line_coordination_x2 = None
lane_separation_line_coordination_y2 = None


# Image dimensions
x_total = None
y_total = None

# time period to throw a warning if no new images found
time_threshold = None

# car counting line coordinates
counting_line = None

# traffic lane separation line coordinates
lane_separation_line = None


class Model:
    def __init__(self, model_file_path):
        gpus = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(gpus[gpu_number], 'CPU')
        #tf.config.set_logical_device_configuration(gpus[gpu_number],[tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit)])

        with tf.device("CPU:{0}".format(gpu_number)):
            self.graph_def = tensorflow.compat.v1.GraphDef()
            self.graph_def.ParseFromString(model_file_path.read_bytes())
            input_names, self.output_names = self._get_graph_inout(self.graph_def)

        # import graph_def
##        with tf.Graph().as_default() as graph:
##            tf.import_graph_def(self.graph_def)

        # print operations
##        for op in graph.get_operations():
##            print(op.name)
       
        self.input_name = input_names[0]
        self.output_names = ["detection_boxes", "detection_classes", "detection_scores"]

        with tf.device("CPU:{0}".format(gpu_number)):
            #self.input_shape = self._get_input_shape(self.graph_def, self.input_name)
            self.input_shape = [600,600]
            tensorflow.import_graph_def(self.graph_def, name = "")

    def predict(self, image_filepath):
        gpus = tf.config.list_physical_devices('CPU')
        tf.config.set_visible_devices(gpus[gpu_number], 'CPU')
        #tf.config.set_logical_device_configuration(gpus[gpu_number],[tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit)])

        with tf.device("CPU:{0}".format(gpu_number)):
            image = PIL.Image.open(image_filepath).resize(self.input_shape)
            input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
            input_array = input_array[:, :, :, (2, 1, 0)]  # => BGR

        with tf.compat.v1.Session() as sess:
            out_tensors = [sess.graph.get_tensor_by_name(o + ":0") for o in self.output_names]
            outputs = sess.run(out_tensors, {self.input_name + ":0": input_array})
            
        return {name: outputs[i][np.newaxis, ...] for i, name in enumerate(self.output_names)}

    @staticmethod
    def _get_graph_inout(graph_def):
        input_names = []
        inputs_set = set()
        outputs_set = set()

        for node in graph_def.node:
            if node.op == "Placeholder":
                input_names.append(node.name)

            for i in node.input:
                inputs_set.add(i.split(':')[0])
            outputs_set.add(node.name)

        output_names = list(outputs_set - inputs_set)
        return input_names, output_names

    @staticmethod
    def _get_input_shape(graph_def, input_name):
        for node in graph_def.node:
            if node.name == input_name:
                return [dim.size for dim in node.attr["shape"].shape.dim][1:3]

    
def has_any_sub_directory(dir_path):
    for fname in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, fname)):
            return True
        
    return False


def update_dirs(root, dirs):
    if(len(dirs) == 0):
        return []
    
    if(len(dirs) == 1):
        return dirs
    
    latest_dir_index = None
    latest_time = None
    
    for dir_index in range(0, len(dirs)):
        if (latest_time is None) or (os.path.getmtime(os.path.join(root, dirs[dir_index])) > latest_time):
            latest_dir_index = dir_index
            latest_time = os.path.getmtime(os.path.join(root, dirs[dir_index]))
          

    if has_any_sub_directory(os.path.join(root, dirs[latest_dir_index])):
        return [dirs[latest_dir_index]]
    
    second_latest_dir_index = None
    second_latest_time = None
    
    for dir_index in range(0, len(dirs)):
        if dir_index == latest_dir_index:
            continue
            
        if (second_latest_time is None) or (os.path.getmtime(os.path.join(root, dirs[dir_index])) > second_latest_time):
            second_latest_dir_index = dir_index
            second_latest_time = os.path.getmtime(os.path.join(root, dirs[dir_index]))                                
    
    return [dirs[second_latest_dir_index], dirs[latest_dir_index]]


def find_recently_added_files(nvr_path, last_check_time):
    new_files = []
    
    for root, dirs, files in os.walk(nvr_path):
        dirs[:] = update_dirs(root, dirs)
        for file in files:
            created_time = os.path.getctime(os.path.join(root, file))
            if file.endswith(".jpg") and created_time > last_check_time:
                new_files.append((os.path.join(root,file), created_time))
                
    return new_files


def print_outputs(outputs):
    assert set(outputs.keys()) == set(["detection_boxes", "detection_classes", "detection_scores"])
    all_boxes =[]
##    print((outputs["detection_boxes"][0][0]).tolist())
##    print((outputs["detection_scores"][0][0]).tolist())
##    print((outputs["detection_classes"][0][0]).tolist())
    for i in range(0, outputs["detection_boxes"][0][0].shape[1] + 1):
        for box, class_id, score in zip(((outputs["detection_boxes"][0][0]).tolist()), ((outputs["detection_classes"][0][0]).tolist()), ((outputs["detection_scores"][0][0]).tolist())):
            if score > probability_threshold:
                all_boxes.append({"Label": class_id, "Probability": score, "box": [box[0], box[1], box[2], box[3]]})
                logger.info(f"Label {class_id} || Probability: {score} || Box: {box[0]} . {box[1]} . {box[2]} . {box[3]}")
        return all_boxes

        
def get_rectangle(left, top, right, bottom):
    return ((left, top), (right, bottom))


def load_model():
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

# Testing slant lane separation lines
def is_above_line(px, py, x1, y1, x2, y2):
    '''
    - Returns True if point (px, py) is above the line (left/right lane depending on indoor/outdoor lots)
    - px, py represents centroid's xy coordinates
    - x1, y1, x2, y2 represent lane separation line coordinates
    '''
    
    # Compute the cross product of lane separation vector and vector between initial lane sep. coordinates and centroid position
    cross_product = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    # If cross product is positive, this indicates that the centroid is positioned above the line.
    
    return cross_product > 0



def car_counter(model, all_frames, car_counter_variables):
    skipper = 1
    if skip_image_counter is not None:
        skipper = skip_image_counter + 1
    
    if output_images_path is not None:
        directory = output_images_path + "\\_output_images\\"
        if not os.path.exists(directory):
            os.makedirs(directory)

    deepsort = DeepSort(n_init=3, max_age=5)
    tracked_cars = {}
    num_cars_in = 0
    num_cars_out = 0
    before = time.time()

    # loop through all images retrieved from NVR
    for j in range(0, len(all_frames), skipper):
        frame = all_frames[j] # single image to process
        outputs = model.predict(pathlib.Path(frame)) # send image to Tensorflow for object detection
        json_output = print_outputs(outputs) # filter out objects not meeting probability threshold
        img = Image.open(pathlib.Path(frame))
        draw = ImageDraw.Draw(img)
        json_output_len = len(json_output)

        # set baseline distance between threshold line and centroid (calculation window)
        min_calc_window = counting_line_coordination_x1 - minimum_calculation_window
        max_calc_window = counting_line_coordination_x1 + maximum_calculation_window
        # Draw predefined lines on the image: calc_window, lane_separation and counting lines
        if output_images_path is not None:
            logger.info("Image {0}".format(frame))
            draw.line(counting_line, fill=(0, 255, 0), width=5)
            draw.line(lane_separation_line, fill=(0, 255, 0), width=1)
            # draw calculation window left
            draw.line((min_calc_window, counting_line_coordination_y1, min_calc_window, counting_line_coordination_y2), fill = (50, 241, 255), width = 5)
            # draw calculation window right
            draw.line((max_calc_window, counting_line_coordination_y1, max_calc_window, counting_line_coordination_y2), fill = (50, 241, 255), width = 5)

        

        # Process detections and prepare them for tracking in format: ( [left,top,w,h] , confidence, detection_class)
        raw_detections = []

        for i in range(0, json_output_len):
            if str(json_output) != "None" :
                if json_output[i]["Label"] == 3.0:
                    # calculate the corners of the box around the object
                    x_min = x_total * json_output[i]["box"][1]
                    x_max = x_total * json_output[i]["box"][3]
                    y_min = y_total * json_output[i]["box"][0]
                    y_max = y_total * json_output[i]["box"][2]

                    bbox = [x_min, y_min, (json_output[i]["box"][3] - json_output[i]["box"][1]) * x_total - 5,
                            (json_output[i]["box"][2] - json_output[i]["box"][0]) * y_total - 5]
                    score = json_output[i]["Probability"]
##                  print(bbox)

                    height = (json_output[i]["box"][3] - json_output[i]["box"][1]) * y_total
                    width = (json_output[i]["box"][2] - json_output[i]["box"][0]) * x_total
               
                    if  width < 60:
                        continue
                    
                    raw_detections.append((bbox, score, 'car'))

                    # determine centroid position
                    centroid_x = (int(x_max) + int(x_min)) / 2
                    centroid_y = (int(y_max) + int(y_min)) / 2
                    draw.ellipse((centroid_x, centroid_y, centroid_x + 5, centroid_y + 5), fill = 'blue')

                    # draw red rectangle around object
                    draw.rectangle(get_rectangle(x_min, y_min, x_max, y_max), outline = "red")
                    # draw box identifier by Tensorflow
                    draw.text((centroid_x-30, centroid_y-40),str("Box: {0}".format(i)), fill = "red", font = ImageFont.truetype("arial", 30))
                    # draw object identifier (car, people, golftcart)
                    draw.text((centroid_x, centroid_y+20),str(json_output[i]["Label"]), fill = "red", font = ImageFont.truetype("arial", 30))

    
        # Convert image from PIL format to a NumPy array
        frame_bgr = np.array(img)
        
        # Update the tracker with raw detections and the current frame, receiving updated track information
        tracks = deepsort.update_tracks(raw_detections, frame=frame_bgr)

        # Iterate through each track to process and count vehicles
        for track in tracks:
            
            # Skip the track if it is not confirmed or hasn't been updated recently
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            
            # Calculate the bounding box's centroid for the current track; orig= True or False, false by default
            bbox = track.to_ltrb()
                
            id = track.track_id
            centroid_x = (bbox[0] + bbox[2]) / 2
            centroid_y = (bbox[1] + bbox[3]) / 2
            
            # Object's centroid must be within calculation window to be tracked
            if min_calc_window <= centroid_y <= max_calc_window:
                if id not in tracked_cars:
                    tracked_cars[id] = {'prev_pos': None, 'curr_pos': (centroid_x, centroid_y)}
                else:
                    tracked_cars[id]['prev_pos'] = tracked_cars[id]['curr_pos']
                    tracked_cars[id]['curr_pos'] = (centroid_x, centroid_y)

                prev_pos = tracked_cars[id]['prev_pos']
                curr_pos = tracked_cars[id]['curr_pos']

                if prev_pos and curr_pos:
                    prev_y = prev_pos[1]
                    curr_y = curr_pos[1]

                    # Increment counters based on the direction of crossing relative to the counting line
                    if prev_y < counting_line_coordination_y1 <= curr_y:
                        num_cars_in += 1
                    elif prev_y > counting_line_coordination_y1 >= curr_y:
                        num_cars_out += 1
                    
            draw.text((centroid_x, centroid_y), str(id), fill="yellow")

        if output_images_path is not None:
            draw.text((20, 70), str(num_cars_out), fill="yellow", font=ImageFont.truetype("arial", 30))
            draw.text((20, 600), str(num_cars_in), fill="blue", font=ImageFont.truetype("arial", 30))
            output_file_name = directory + os.path.basename(frame)
            img.save(output_file_name, "JPEG")
            logger.info(f"Last Image Detected: {output_file_name}.JPEG")

    diff = time.time() - before
    logger.info(f"Time to process {len(all_frames)} images: {diff} seconds")
    
    return (num_cars_in, num_cars_out)


def validate_arguments():
    
    if model_file_path is None:
        logger.error("--model-file-path is missing")
        raise ValueError("--model-file-path is missing")
        
    if probability_threshold is None:
        logger.error("--probability-threshold is missing")
        raise ValueError("--probability-threshold is missing")
        
    if counting_line_coordination_x1 is None:
        logger.error("--counting-line-coordination-x1 is missing")
        raise ValueError("--counting-line-coordination-x1 is missing")
        
    if counting_line_coordination_y1 is None:
        logger.error("--counting-line-coordination-y1 is missing")
        raise ValueError("--counting-line-coordination-y1 is missing")
        
    if counting_line_coordination_x2 is None:
        logger.error("--counting-line-coordination-x2 is missing")
        raise ValueError("--counting-line-coordination-x2 is missing")
        
    if counting_line_coordination_y2 is None:
        logger.error("--counting-line-coordination-y2 is missing")
        raise ValueError("--counting-line-coordination-y2 is missing")

    if lane_separation_line_coordination_x1 is None:
        logger.error("--lane-separation-line-coordination-x1 is missing")
        raise ValueError("--lane-separation-line-coordination-x1 is missing")
        
    if lane_separation_line_coordination_y1 is None:
        logger.error("--lane-separation-line-coordination-y1 is missing")
        raise ValueError("--lane-separation-line-coordination-y1 is missing")
        
    if lane_separation_line_coordination_x2 is None:
        logger.error("--lane-separation-line-coordination-x2 is missing")
        raise ValueError("--lane-separation-line-coordination-x2 is missing")
        
    if lane_separation_line_coordination_y2 is None:
        logger.error("--lane-separation-line-coordination-y2 is missing")
        raise ValueError("--lane-separation-line-coordination-y2 is missing")

    if x_total is None:
        logger.error("--x-total is missing")
        raise ValueError("--x-total is missing")
        
    if y_total is None:
        logger.error("--y-total is missing")
        raise ValueError("--y-total is missing")
        
    if time_threshold is None:
        logger.error("--time-threshold is missing")
        raise ValueError("--time-threshold is missing")
        
    if zone_number is None:
        logger.error("--zone-number is missing")
        raise ValueError("--zone-number is missing")
        
    if base_url is None:
        logger.error("--base-url is missing")
        raise ValueError("--base-url is missing")
        
    if nvr_path is None:
        logger.error("--nvr-path is missing")
        raise ValueError("--nvr-path is missing")
        
    if car_counter_variables_path is None:
        logger.error("--car-counter-variables-path is missing")
        raise ValueError("--car-counter-variables-path is missing")

    if gpu_number is None:
        logger.error("--gpu-number is missing")
        raise ValueError("--gpu-number is missing")

    if minimum_calculation_window is None:
        logger.error("--minimum-calculation-window is missing")
        raise ValueError("--minimum-calculation-window is missing")

    if maximum_calculation_window is None:
        logger.error("--maximum-calculation-window is missing")
        raise ValueError("--maximum-calculation-window is missing")
         
def set_global_variables(argv):
    global model_file_path
    global probability_threshold
    global counting_line
    global lane_separation_line
    global counting_line_coordination_x1
    global counting_line_coordination_y1
    global counting_line_coordination_x2
    global counting_line_coordination_y2
    global lane_separation_line_coordination_x1 
    global lane_separation_line_coordination_y1
    global lane_separation_line_coordination_x2
    global lane_separation_line_coordination_y2
    global x_total
    global y_total
    global time_threshold
    global zone_number
    global parking_lot_type
    global traffic_flow_direction
    global base_url
    global nvr_path
    global car_counter_variables_path
    global output_images_path
    global log_path
    global log_level
    global skip_image_counter
    global gpu_number
    global gpu_mem_limit
    global minimum_calculation_window
    global maximum_calculation_window
   
    usage = "<script-name> -m <model-file-path> -p <probability-threshold> -a <counting-line-coordination-x1> -b <counting-line-coordination-y1> -c <counting-line-coordination-x2> -e <counting-line-coordination-y2> -s <lane-separation-line-coordination-x1> -k <lane-separation-line-coordination-y1> -m <lane-separation-line-coordination-x2> -n <lane-separation-line-coordination-y2> -d <x-total> -z <y-total> -t <time-threshold> -z <zone-number> -l <parking-lot-type> -f <traffic-flow-direction> -u <base-url> -n <nvr-path> -g <car-counter-variables-path> -o <ouput-images-path> -i <log-path> -x <log-level> -y <skip-image-counter> -gn <gpu-number> -gm <gpu-memory-limit> -mincw <minimum-calculation-window> -maxcw <maximum-calculation-window>"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:p:a:b:c:e:s:k:m:n:d:z:t:z:l:f:u:n:g:o:i:x:y:gn:gm:mincw:maxcw", ["model-file-path=",
                                                                                                                     "probability-threshold=", 
                                                                                                                     "counting-line-coordination-x1=", 
                                                                                                                     "counting-line-coordination-y1=",
                                                                                                                     "counting-line-coordination-x2=",
                                                                                                                     "counting-line-coordination-y2=",
                                                                                                                     "lane-separation-line-coordination-x1=",
                                                                                                                     "lane-separation-line-coordination-y1=",
                                                                                                                     "lane-separation-line-coordination-x2=",
                                                                                                                     "lane-separation-line-coordination-y2=",
                                                                                                                     "x-total=",
                                                                                                                     "y-total=",
                                                                                                                     "time-threshold=",
                                                                                                                     "zone-number=",
                                                                                                                     "parking-lot-type=",
                                                                                                                     "traffic-flow-direction=",
                                                                                                                     "base-url=",
                                                                                                                     "nvr-path=",
                                                                                                                     "car-counter-variables-path=", 
                                                                                                                     "output-images-path=",
                                                                                                                     "log-path=",
                                                                                                                     "log-level=",
                                                                                                                     "skip-image-counter=",
                                                                                                                     "gpu-number=",
                                                                                                                     "gpu-memory-limit=",
                                                                                                                     "minimum-calculation-window=",
                                                                                                                     "maximum-calculation-window="])
        
        
    except getopt.GetoptError:
        raise ValueError(usage)

    for opt, arg in opts:
        if opt in ("-m", "--model-file-path"):
            model_file_path = arg
            
        elif opt in ("-p", "--probability-threshold"):
            probability_threshold = float(arg)
                   
        elif opt in ("-a", "--counting-line-coordination-x1"):
            counting_line_coordination_x1 = int(arg)
       
        elif opt in ("-b", "--counting-line-coordination-y1"):
            counting_line_coordination_y1 = int(arg)
        
        elif opt in ("-c", "--counting-line-coordination-x2"):
            counting_line_coordination_x2 = int(arg)
        
        elif opt in ("-e", "--counting-line-coordination-y2"):
            counting_line_coordination_y2 = int(arg)
                   
        elif opt in ("-s", "--lane-separation-line-coordination-x1"):
            lane_separation_line_coordination_x1 = int(arg)
            
        elif opt in ("-k", "--lane-separation-line-coordination-y1"):
            lane_separation_line_coordination_y1 = int(arg)
            
        elif opt in ("-m", "--lane-separation-line-coordination-x2"):
            lane_separation_line_coordination_x2 = int(arg)
            
        elif opt in ("-n", "--lane-separation-line-coordination-y2"):
            lane_separation_line_coordination_y2 = int(arg)
                   
        elif opt in ("-d", "--x-total"):
            x_total = int(arg)
            
        elif opt in ("-z", "--y-total"):
            y_total = int(arg)
           
        elif opt in ("-t", "--time-threshold"):
            time_threshold = int(arg)
        
        elif opt in ("z", "--zone-number"):
            zone_number = arg
        
        elif opt in ("l", "--parking-lot-type"):
            parking_lot_type = arg
         
        elif opt in ("f", "--traffic-flow-direction"):
            traffic_flow_direction = arg
            
        elif opt in ("u", "--base-url"):
            base_url = arg
            
        elif opt in ("n", "--nvr-path"):
            nvr_path = arg
            
        elif opt in ("g", "--car-counter-variables-path"):
            car_counter_variables_path = arg
            
        elif opt in ("o", "--output-images-path"):
            output_images_path = arg
           
        elif opt in ("i", "--log-path"):
            log_path = arg
            
        elif opt in("x", "--log-level"):
            log_level = arg
                                                                                                   
        elif opt in ("y", "--skip-image-counter"):
            skip_image_counter = int(arg)

        elif opt in ("-gn", "--gpu-number"):
            gpu_number = int(arg)

        elif opt in ("-gm", "--gpu-memory-limit"):
            gpu_mem_limit = int(arg)

        elif opt in ("-mincw", "--minimum-calculation-window"):
            minimum_calculation_window = int(arg)

        elif opt in ("-maxcw", "--maximum-calculation-window"):
            maximum_calculation_window = int(arg)
            
    validate_arguments()
    
    # car counting line coordinates
    counting_line = (counting_line_coordination_x1, 
                     counting_line_coordination_y1,
                     counting_line_coordination_x2,
                     counting_line_coordination_y2)
    
    
    # traffic lane separation line coordinates
    lane_separation_line = (lane_separation_line_coordination_x1,
                            lane_separation_line_coordination_y1,
                            lane_separation_line_coordination_x2,
                            lane_separation_line_coordination_y2)



def send_put_request(url, num_cars_in, num_cars_out):
    #secrets = dotenv_values(".env")
    #headers = {"Content-Type": "application/json","x-functions-key": secrets["API_KEY"]}
    
    data = {"countIn": num_cars_in, "countOut": num_cars_out}
    #try:
    #    response = requests.put(url, data=json.dumps(data), headers=headers)
    #except Exception as e:
    #    logger.error(f"Exception {str(e)} when submitting car count")
   
def read_in_car_counter_variables():
    try:
        with open(car_counter_variables_path) as f:
            car_counter_variables = json.load(f)
            return car_counter_variables
    except Exception as e:
        logger.info(f"Exception {str(e)} resides in read_in_car_counter_variables")
        

def write_out_car_counter_variables(car_counter_variables):
    try:
        with open(car_counter_variables_path, 'w') as f:
            json_object = json.dumps(car_counter_variables, indent = 2)
            f.write(json_object)
    except Exception as e:
        logger.info(f"Exception {str(e)} resides in write_out_car_counter_variables")
        
def set_log_level():
    if log_level is None:
        logger.setLevel(logging.NOTSET)
        return 
    
    level = log_level.strip().upper()
    level = level.replace("'", "")
    
    if(level == "CRITICAL"):
        logger.setLevel(logging.CRITICAL)
        
    elif(level == "ERROR"):
        logger.setLevel(logging.ERROR)
        
    elif(level == "WARNING"):
        logger.setLevel(logging.WARNING)
        
    elif(level == 'INFO'):
        logger.setLevel(logging.INFO)
        
    elif(level == "DEBUG"):
        logger.setLevel(logging.DEBUG)
              
        
def setup_logger():
    global logger

    if log_path is not None:
        log_file_name = log_path + "\\car_counter_logger.log"
        if not os.path.exists(log_path): 
            os.makedirs(log_path)
        if not os.path.isfile(log_file_name):
            log_file = open(log_file_name, 'w+')
        logging.basicConfig(filename = log_file_name, format ='%(asctime)s | %(levelname)s | %(name)s | %(message)s', filemode = 'a') 
        logger = logging.getLogger()
        set_log_level()
        logger.disabled = False
        
    else:
        logger = logging.getLogger()
        logger.disabled = True
    

def main(argv):
    set_global_variables(sys.argv)
    setup_logger()

    model = Model(pathlib.Path(model_file_path))

    url = base_url + '/zones/' + zone_number + '/count'
    
    car_counter_variables = read_in_car_counter_variables()
   
    new_files = find_recently_added_files(nvr_path, car_counter_variables["last_check_time"])
    car_counter_variables["last_check_time"] = car_counter_variables["last_check_time"] if not new_files else new_files[-1][1]
    all_frames = [file[0] for file in new_files]
    
    if len(all_frames) > 0:
        last_time_new_file_added = time.time()
        (num_cars_in, num_cars_out) = car_counter(model, all_frames, car_counter_variables)
        logger.info("Persisting the number of cars in/out into database ...")
        send_put_request(url, num_cars_in, num_cars_out)
        logger.info("Successfully persisted the number of cars in/out into database")
        write_out_car_counter_variables(car_counter_variables)
            
    else:
        write_out_car_counter_variables(car_counter_variables)
        now = time.time()
        if (car_counter_variables["last_time_new_file_added"] != -1) and (now - car_counter_variables["last_time_new_file_added"] > time_threshold):
                logger.error("No new images has been detected for a while!")
                raise NameError("No new images has been detected for a while!")

        
if __name__ == '__main__':
    main(sys.argv)
