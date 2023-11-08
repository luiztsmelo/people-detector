import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as cv from 'opencv4nodejs';

async function detectPeopleInVideo() {
  // Load the Coco SSD model
  const model = await cocoSsd.load();

  // Create a video capture object to read the RTSP stream
  const videoCapture = new cv.VideoCapture('rtsp://zephyr.rtsp.stream/movie?streamKey=83be864457e2277c948f010eed7004de');

  // Read frames from the video stream
  let frame;
  while ((frame = videoCapture.read())) {
    // Convert the frame to a TensorFlow tensor
    const tensor = tf.tensor(frame.getData(), [frame.rows, frame.cols, 3]);

    // Run object detection on the frame
    const predictions = await model.detect(tensor);

    // Count the number of people detected
    const numPeople = predictions.filter((prediction) => prediction.class === 'person').length;

    console.log(`Number of people detected: ${numPeople}`);

    // Display the frame with bounding boxes around the detected objects
    const image = cv.cvtColor(frame, cv.COLOR_BGR2RGB);
    predictions.forEach((prediction) => {
      const { x, y, width, height } = prediction.bbox;
      const rect = new cv.Rect(x, y, width, height);
      image.drawRectangle(rect, new cv.Vec(0, 255, 0), 2);
    });
    cv.imshow('Video', image);
    cv.waitKey(1);
  }

  // Release the video capture object
  videoCapture.release();
}

detectPeopleInVideo();
