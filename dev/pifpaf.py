import cv2
import openpifpaf

capture = cv2.VideoCapture('.webm')
_, image = capture.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
predictions, gt_anns, meta = predictor.numpy_image(image)

annotation_painter = openpifpaf.show.AnnotationPainter()
with openpifpaf.show.Canvas.image(image) as ax:
    annotation_painter.annotations(ax, predictions)