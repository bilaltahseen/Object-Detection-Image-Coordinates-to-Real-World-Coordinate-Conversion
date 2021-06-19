# Object Detection & Image Coordinates to Real World Coordinate Conversion

Using tensorflow object detection api and openCV to calculate real world
coordinates from top view with fixed height of the camera.

## Installation

```bash
pip install tensorflow==2.4.0
pip install opencv-python
git clone https://github.com/tensorflow/models.git
# Install tensorflow object detection api from models
# After clonning models folder copy \models\research\object_detection\packages\tf2\setup.py
# Paste setup.py at \models\research
python setup.py install
```

## Objective

- Train a custom object detection classifier.
- Detect bounding boxes of object.
- Calculate bounding boxes centers.
- Convert image coordinates to real world coordinates.

## Implementation For Coordinates Conversion

- First we will find pixel to cm ratio which is given by width of the image frame divided by the actual cm units visible in the whole frame

```python
px_to_cm_ration = 22/640.0
```

- Calculate the homogenous matrix to convert the rotation and distance from
  camera cordinates to real world.

```python
Rad = (95.0/180.0) * np.pi
    RZ = [[np.cos(Rad), -np.sin(Rad), 0],
          [np.sin(Rad), np.cos(Rad), 0], [0, 0, 1]]
    R180_X = [[1, 0, 0], [
        0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]]
    RO_C = np.dot(RZ, R180_X)
    DO_C = [[-8.5], [-22], [0]] # Distance Matrix
    HO_C = np.concatenate((RO_C, DO_C), 1)
    HO_C = np.concatenate((HO_C, [[0, 0, 0, 1]]), 0)
    px_to_cm_ration = 22/640.0
    X_C = x_loc*px_to_cm_ration
    Y_C = y_loc*px_to_cm_ration
    PC = [[X_C], [Y_C], [0], [1]]
    PO = np.dot(HO_C, PC)
```

[Conversion Tutorial](https://www.youtube.com/watch?v=kV9VlHxQwNQ&t=329s)

## Testing

![testing image](test.png)

```python
print('Object 1 (x,y)', '7.3', '14.6')
```
