import 'package:flutter/services.dart' show rootBundle;

import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

// Future<List<TensorImage>> getHummingbirdImages() async {
//   List<TensorImage> inputImages = [];
//   for (int i = 0; i <= 259; i++) {
//     String imageNum = i.toString().padLeft(3, '0');
//     String imageFile = 'assets/hummingbird/frame_${imageNum}_delay-0.04s.png';
//     img.Image? image = img
//         .decodeImage((await rootBundle.load(imageFile)).buffer.asUint8List());
//     if (image == null) {
//       continue;
//     }
//     TensorImage tIm = TensorImage.fromImage(image);
//     inputImages.add(tIm);
//   }
//   return inputImages;
// }

// List<TensorImage> preprocessImages(List<TensorImage> images) {
//   double imageMean = 127.5;
//   double imageStd = 127.5;
//   int imageSize = 224;

//   ImageProcessor imageProcessor = ImageProcessorBuilder()
//       .add(CastOp(tfl.TfLiteType.float32))
//       .add(NormalizeOp(imageMean, imageStd))
//       .add(ResizeOp(imageSize, imageSize, ResizeMethod.BILINEAR))
//       .build();

//   List<TensorImage> preprocessedImages = [];

//   for (TensorImage image in images) {
//     TensorImage processedImage = imageProcessor.process(image);
//     preprocessedImages.add(processedImage);
//   }

//   return preprocessedImages;
// }

// List<int> clipByValue(List<int> idxes, int min, int max) {
//   for (int i = 0; i < idxes.length; i++) {
//     if (idxes[i] < min) {
//       idxes[i] = min;
//     } else if (idxes[i] > max) {
//       idxes[i] = max;
//     }
//   }

//   return idxes;
// }

Future<int> getCounts(tfl.Interpreter interpreter) async {
  // List<TensorImage> images = await getHummingbirdImages();
  // images = preprocessImages(images);
  // int sequenceLength = images.length;
  // List rawScoresList = [];
  // List scores = [];
  // List withinPeriodScoresList = [];

  List<int> strides = [1, 2, 3, 4];
  // int batchSize = 1;
  // double THRESHOLD = 0.2;
  // double WITHIN_PERIOD_THRESHOLD = 0.5;

  // int modelNumFrames = 64;
  // int modelImageSize = 112;

  for (int stride in strides) {
    // int numBatches =
    // (sequenceLength / modelNumFrames / stride / batchSize).ceil();
    //   List<double> rawScoresPerStride = [];
    //   List<double> withinPeriodScoresPerStride = [];
    int numBatches = 1;
    for (int batchNum = 0; batchNum < numBatches; batchNum++) {
      //     int start = batchNum * batchSize * modelNumFrames * stride;
      //     int end = (batchNum + 1) * batchSize * modelNumFrames * stride;

      //     List<int> idxes = [for (int i = start; i < end; i += stride) i];
      //     idxes = clipByValue(idxes, 0, sequenceLength - 1);

      //     // gather frames for the batch
      //     List<TensorBuffer> batchImages = [];
      //     for (int i = 0; i < idxes.length; i++) {
      //       TensorBuffer image = images[idxes[i]].getTensorBuffer();
      //       batchImages.add(image);
      //     }

      List<int> inputShape = interpreter.getInputTensor(0).shape;
      tfl.TfLiteType inputType = interpreter.getInputTensor(0).type;

      // List<List<int>> outputDims = [];
      // for (tfl.Tensor outputTensor in interpreter.getOutputTensors() ) {
      //     List<int> shape = outputTensor.shape;
      //     outputDims.add(shape);
      // }

      // tfl.TfLiteType outputType = interpreter.getOutputTensor(0).type;
      // TensorBuffer outputBuffer =
      //     TensorBuffer.createFixedSize(outputDims, outputType);

      // inputBuffer.loadList(listImgs, shape: inputBuffer.getShape());
      // List<List<TensorBuffer>> listImgs = [batchImages];

      // run inference
      List rawScores = List<double>.filled(1 * 64 * 32, 0).reshape([1, 64, 32]);
      List withinPeriodScores =
          List<double>.filled(1 * 64 * 1, 0).reshape([1, 64, 1]);
      List periodScores =
          List<double>.filled(1 * 64 * 512, 0).reshape([1, 64, 512]);
      var outputBuffer = [rawScores, withinPeriodScores, periodScores];

      var inShape2 = interpreter.getInputTensor(0).shape;
      var inType2 = interpreter.getInputTensor(0).type;
      var inputBuffer = TensorBuffer.createFixedSize(inShape2, inType2);

      interpreter.run(inputBuffer, outputBuffer);

      print(outputBuffer);
      return 0;
    }
  }
  return 0;
}

void runRepnet() async {
  tfl.Interpreter interpreter =
      await tfl.Interpreter.fromAsset("repnet2.5.tflite");

  interpreter.allocateTensors();
  var input = interpreter.getInputTensors()[0];
  print("Input shape: ${input.shape}");
  print("Interpreter created successfully");

  int counts = await getCounts(interpreter);

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
