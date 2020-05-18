# Facial Expression Recognition
Facial Expression Recognition in Keras using a CNN.

![Demo](https://raw.githubusercontent.com/shaswata56/Facial_Expression_Recognition/master/res/test.gif)

![Accuracy vs. Loss](https://raw.githubusercontent.com/shaswata56/Facial_Expression_Recognition/master/res/plot.png)

can be done realtime from webcam by changing,

```python3
  self.video = cv2.VideoCapture("videos/presidential_debate.mp4")
```
to
```python3
  self.video = cv2.VideoCapture(0)
```
