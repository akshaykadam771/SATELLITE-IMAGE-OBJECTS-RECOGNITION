# importing the necessary dependencies
from flask import Flask, render_template, request,send_file
from flask_cors import CORS,cross_origin
import numpy as np
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import shutil




from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__) # initializing a flask app
CORS(app)

MODEL_NAME = 'my_model_satellite_detection_mask_rcnn'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'labelmapsat.pbtxt')

# loading tf graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# loading classes file
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    
    return render_template("index.html")

@app.route('/startapp',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def upload_img():
    if request.method == 'POST':
        return render_template('upload.html')

    else:
        return "something went wrong"



@app.route('/detection',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def detection():
    if request.method == 'POST':
        try:
            #reading image file
            uploaded_file = request.files['upload_file']
            filename = uploaded_file.filename

            #procede only if file is available
            if uploaded_file.filename != '':
                uploaded_file.save(filename)



                #procede only if image is in "jpg,jpeg,png" format
                image_file_name=str(filename)
                name_split = image_file_name.split(".")
                extension = name_split[-1]
                extension = extension.upper()
                allowed_extensions = ["JPG","JPEG"]
                proceed = "False"
                for i in allowed_extensions:
                    if (i == extension):
                        proceed = "True"
                        print("Extension Exists")

                # procede only if for allowed extension of image file
                if proceed == "True":

                    #saving input image to test_images folder
                    try:
                        target = './' + str(filename)
                        destination = "./test_images/"
                        shutil.copy(target, destination)
                    except Exception as e:
                        return "file saving error in folder"

                    #loading image
                    def load_image_into_numpy_array(image):
                        (im_width, im_height) = image.size
                        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

                    try:
                        PATH_TO_TEST_IMAGES_DIR = "test_images"
                        TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename)]

                        # Size, in inches, of the output images.
                        IMAGE_SIZE = (12, 8)

                    except Exception as e:
                        print('Error with image file loading properties',e)


                    try:
                        def run_inference_for_single_image(image, graph):
                            with graph.as_default():
                                with tf.Session() as sess:

                                    # Get handles to input and output tensors
                                    ops = tf.get_default_graph().get_operations()
                                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                                    tensor_dict = {}
                                    for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                                        tensor_name = key + ':0'
                                        if tensor_name in all_tensor_names:
                                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                                    if 'detection_masks' in tensor_dict:
                                        # The following processing is only for single imag
                                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks,detection_boxes, image.shape[0], image.shape[1])
                                        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                                        # Follow the convention by adding back the batch dimension
                                        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                                    # Run inference
                                    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

                                    # all outputs are float32 numpy arrays, so convert types as appropriate
                                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                                    if 'detection_masks' in output_dict:
                                        output_dict['detection_masks'] = output_dict['detection_masks'][0]

                            return output_dict

                    except Exception as e:
                        return "Something Went Wrong....Unable Please try again."


                    # Now start detection with passing the image
                    try:
                        for image_path in TEST_IMAGE_PATHS:
                            image = Image.open(image_path)

                            # result image with boxes and labels on it.
                            image_np = load_image_into_numpy_array(image)

                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)

                            # Actual detection.
                            output_dict = run_inference_for_single_image(image_np, detection_graph)

                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)
                            plt.figure(figsize=IMAGE_SIZE)
                            plt.imshow(image_np)
                            plt.savefig("masked.png")


                    except Exception as e:
                            return "Something Went Wrong....Unable to do detect Please try again."

                    # copy masked image into static folder
                    try:
                        target1 = './masked.png'
                        destination1 = "./static/masked_image"
                        shutil.copy(target1, destination1)
                    except Exception as e:
                        return "file saving error in folder"

                    # Extracting classes and scores from detection with minumum threshold=70%
                    try:
                        classes = output_dict['detection_classes']
                        scores = output_dict['detection_scores']
                        min_score_thresh = 0.7
                        dicta = {}
                        for j in range(0, 3):
                            if scores[j] >= min_score_thresh:
                                dect_class = classes[j]
                                dect_score = scores[j]
                                dect_score = "{:.2f}".format(dect_score)
                                dicta.update({dect_class: dect_score})

                        cls = dicta.keys()
                        cls = list(cls)
                        scr = dicta.values()
                        scr = list(scr)

                        for i in range(len(dicta)):
                            if cls[i] == 1:
                                scr1 = scr[i]
                                scr1 = float(scr1) * 100
                                out1 = "RIVER (" + str(scr1) + ")% CONFIDENCE"

                            if cls[i] == 2:
                                scr1 = scr[i]
                                scr1 = float(scr1) * 100
                                out2 = "FOREST (" + str(scr1) + ")% CONFIDENCE"

                            if cls[i] == 3:
                                scr1 = scr[i]
                                scr1 = float(scr1) * 100
                                out3 = "BUILDING (" + str(scr1) + ")% CONFIDENCE"

                        # return string to the show_image.html page
                        if len(dicta) == 1:
                            if cls[0] == 1:
                                out_str = out1

                            if cls[0] == 2:
                                out_str = out2

                            if cls[0] == 3:
                                out_str = out3

                            return render_template('show_image.html', out_str=out_str)

                        if len(dicta) == 2:
                            if (cls[0]==1) & (cls[1]==2):
                                out_str = out1 + " , " + out2

                            if (cls[0]==1) & (cls[1]==3):
                                out_str = out1 + " , " + out3

                            if (cls[0]==2) & (cls[1]==3):
                                out_str = out2 + " , " + out3

                            if (cls[0]==2) & (cls[1]==1):
                                out_str = out1 + " , " + out2

                            if (cls[0]==3) & (cls[1]==1):
                                out_str = out1 + " , " + out3

                            if (cls[0]==3) & (cls[1]==2):
                                out_str = out2 + " , " + out3

                            return render_template('show_image.html', out_str=out_str)

                        if len(dicta) == 3:
                            out_str = out1 + " , " + " , " + out3
                            return render_template('show_image.html', out_str=out_str)


                    except Exception as e:
                        return "Something Went Wrong....Classes and Score extraction error Please try again."

                else:
                    return 'Error: Please Make Sure that image file is in standard acceptable extension,Please go through given Sample image file format'

            else:
                return 'File Not Found'

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

    else:
        return render_template('index.html')



@app.route('/uploadfile',methods=['POST','GET'])  #
@cross_origin()
def uploadfile():
    return render_template('upload.html')




if __name__ == "__main__":
    #to run on cloud
    #port = int(os.getenv("PORT"))
    #app.run(host='0.0.0.0', port=port)  # running the app

    #to run locally
    app.run(host='127.0.0.1', port=8000, debug=True)



