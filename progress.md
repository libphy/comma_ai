### 09/2016
Followed instructions in DriveSim.md.  
It crashed due to memory issue, so I modified to reduce a batch size. It worked but crashed later.

### 10/17/2016
Followes instructions in SelfSteering.md, which seems to be the main challenge of the Udacity event.
It trained without problem (1 overnight), and also the pygame visualisation worked but the visualization was not so useful to quantify how well the prediction follows actual line. One problem when the prediction completely fails is when the car make big turns. Also it seems that the line projection does not care whether other cars change lanes and come very close in front of the car. As for scenes with many cars, it seems that the steering somehow use anchoring to the car in front, but it's unclear how and why it does it. I need to see more on how their model works and what they want to achieve. But anyway, it seems the challenge deadline (end of Oct.) is too short for my time line.    

### 10/18/2016
Reviewed the model used in train_steering_model.py.
It's quite simple model.
The first layer is a simple transformation which I don't know why it's there. But I confirmed that it calculates element-wise calculation of x/127.5 -1. It seems it's rescaling the pixel values.
To check, I fed an arbitrary matrix of the shape (2,3,160,320) to the model loaded from json and keras.
```
from keras import models
with open('./outputs/steering_model/steering_angle.json','r') as f:
   M = json.load(f)  ## this is the json string
Md = json.loads(M) ## this is a dictionary   
model = models.model_from_json(M) # M contains model configuration information
model.load_weights('./outputs/steering_model/steering_angle.keras')
# Once the model is loaded, saved weights can be loaded. Note that it's .keras form not hd5f.

```
Then I pull out the first layer 'Lambda' (index 0) and feed an arbitrary input tensor that matches dimension as defined in the model.
```
from keras import backend as K
a = np.arange(2*3*160*320).reshape((2, 3, 160,320))

In [164]: model.fit(a,[1,1],1,1) # I set the batch size and epoch to be 1, and arbitrary target [1,1] (matching dimension)
Epoch 1/1
2/2 [==============================] - 0s - loss: 945073.6172     
Out[164]: <keras.callbacks.History at 0x7f4730ec4f90>

In [165]: l0 = model.layers[0]

In [166]: inputs = [K.learning_phase()] + model.inputs

In [167]: _out_f = K.function(inputs,[l0.output])

In [168]: out = _out_f([0]+[a])

In [169]: out
```
It's easy to check what the lambda layer does.

### 10/22/2016 - 10/23/2016
Worked on understanding how data is streamed.
In server.py, zmq is used to build sender and receiver.
(Tutorial lecture at zmq website)- zmq is a library (many language versions exist other than python) that makes easy to stream data (originally used for stock market data to send millions of high volume transactions messages between thousands of cores) asynchronously.

server.py uses a keras module called model.fit_generator. fit_generator does some multi-threading stuff(?) using nb_worker, which I don't understand very well, but the core function is actually processing the 'generator' which is passed into fit_generator as an input. In server.py, 'gen' which is a direct input of the function 'start_server' is constructed by a function datagen, which comes from another script called 'dask_generator'. The function 'datagen' extracts data (c5x, angle, speed, filters, hdf5_camera) from an input called 'filter_files'. It seems the actual image data x is wrapped in c5x (see line 107 of 'dask_generatot.py').
I'm puzzled by what .fit_generator does about processing the data that has passed by gen. In the [keras source code](https://github.com/fchollet/keras/blob/master/keras/models.py), class Sequential has a bunch of methods including fit_generator. This method simply returns 'self.model.fit_generator(...)', where 'model' is an internal instance created inside the class. I couldn't see where actual "fitting" is happening. One clue is that class Sequenial inherits the class Model in keras/engine/training.py, where an actual process of training takes place. It's not so clear how, but it seems that the instance model gets trained by things inside Model.     
