import 'dart:io';

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

void main() {
  runApp(const MyApp());
}

List<TensorImage> getHummingbirdImages() {
  List<TensorImage> inputImages = [];
  for (int i = 0; i <= 259; i++) {
    String imageNum = i.toString().padLeft(3, '0');
    File imageFile = File.fromUri(
        Uri.parse('assets/hummingbird/frame_${imageNum}_delay-0.04s.png'));
    TensorImage tensorImage = TensorImage.fromFile(imageFile);
    inputImages.add(tensorImage);
  }
  return inputImages;
}

List<TensorImage> preprocessImages(List<TensorImage> images) {
  double imageMean = 127.5;
  double imageStd = 127.5;
  int imageSize = 224;

  ImageProcessor imageProcessor = ImageProcessorBuilder()
      .add(CastOp(tfl.TfLiteType.float32))
      .add(NormalizeOp(imageMean, imageStd))
      .add(ResizeOp(imageSize, imageSize, ResizeMethod.BILINEAR))
      .build();

  List<TensorImage> preprocessedImages = [];

  for (TensorImage image in images) {
    TensorImage processedImage = imageProcessor.process(image);
    preprocessedImages.add(processedImage);
  }

  return preprocessedImages;
}

List<int> clipByValue(List<int> idxes, int min, int max) {
  for (int i = 0; i < idxes.length; i++) {
    if (idxes[i] < min) {
      idxes[i] = min;
    } else if (idxes[i] > max) {
      idxes[i] = max;
    }
  }

  return idxes;
}

int getCounts(tfl.Interpreter interpreter) {
  List<TensorImage> images = getHummingbirdImages();
  images = preprocessImages(images);
  int sequenceLength = images.length;
  List rawScoresList = [];
  List scores = [];
  List withinPeriodScoresList = [];

  List<int> strides = [1, 2, 3, 4];
  int batchSize = 1;
  double THRESHOLD = 0.2;
  double WITHIN_PERIOD_THRESHOLD = 0.5;

  int modelNumFrames = 64;
  int modelImageSize = 112;

  for (int stride in strides) {
    int numBatches =
        (sequenceLength / modelNumFrames / stride / batchSize).ceil();
    List<double> rawScoresPerStride = [];
    List<double> withinPeriodScoresPerStride = [];

    for (int batchNum = 0; batchNum < numBatches; batchNum++) {
      int start = batchNum * batchSize * modelNumFrames * stride;
      int end = (batchNum + 1) * batchSize * modelNumFrames * stride;

      List<int> idxes = [for (int i = start; i < end; i += stride) i];
      idxes = clipByValue(idxes, 0, sequenceLength - 1);

      // gather frames for the batch
      List<TensorBuffer> batchImages = [];
      for (int i = 0; i < idxes.length; i++) {
        TensorBuffer image = images[idxes[i]].getTensorBuffer();
        batchImages.add(image);
      }

      // run inference
      List rawScores = List<double>.filled(1 * 64 * 32, 0).reshape([1, 64, 32]);
      List withinPeriodScores =
          List<double>.filled(1 * 64 * 1, 0).reshape([1, 64, 1]);
      List periodScores =
          List<double>.filled(1 * 64 * 512, 0).reshape([1, 64, 512]);
      var outputBuffers = [rawScores, withinPeriodScores, periodScores];

      interpreter.run(batchImages, outputBuffers);

      print(outputBuffers);
    }
  }
  return 0;
}

void runRepnet() async {
  // check if the model file exists
  // File modelFile = File('/assets/repnet.tflite');
  // if (!modelFile.existsSync()) {
  //   print('Model file not found');
  //   return;
  // }

  // load file into memory
  tfl.Interpreter interpreter =
      await tfl.Interpreter.fromAsset("repnet2.5.tflite");

  int counts = getCounts(interpreter);

  interpreter.close();
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    runRepnet();
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // Try running your application with "flutter run". You'll see the
        // application has a blue toolbar. Then, without quitting the app, try
        // changing the primarySwatch below to Colors.green and then invoke
        // "hot reload" (press "r" in the console where you ran "flutter run",
        // or simply save your changes to "hot reload" in a Flutter IDE).
        // Notice that the counter didn't reset back to zero; the application
        // is not restarted.
        primarySwatch: Colors.blue,
      ),
      // home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}
