import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
// to use root bundle
import 'package:flutter/services.dart' as services;

void main() {
  runApp(const MyApp());
}

Future<List<services.ByteData>> getHummingbirdImages() async {
  final AssetBundle rootBundle = services.rootBundle;
  List<services.ByteData> inputImages = [];
  for (int i = 0; i <= 259; i++) {
    // int formatted to 3sf
    String imageNum = i.toString().padLeft(3, '0');
    // final image = Image.asset('assets/hummingbird/frame_$imageNum-0.04s.png');
    final imageBytes =
        await rootBundle.load('assets/hummingbird/frame_$imageNum-0.04s.png');
    // convert image to bytes
    inputImages.add(imageBytes);
  }
  return inputImages;
}

List<services.ByteData> preprocessImages(List<services.ByteData images) {
  
}

void getCounts() async {
  List<services.ByteData> images = await getHummingbirdImages();
  int sequenceLength = images.length;
  List rawScoresList = [];
  List scores = [];
  List withinPeriodScoresList = [];

  int modelNumFrames = 64;
  int modelImageSize = 112;


}

void runRepnet() async {
  // final interpreter = await tfl.Interpreter.fromAsset('repnet.tflite');
  // final input = interpreter.getInputTensor(0);
  // final output = interpreter.getOutputTensor(0);

  // ImageProcessor imageProcessor = ImageProcessorBuilder()
  //     .add(ResizeOp(112, 112, ResizeMethod.BILINEAR))
  //     .build();

  // TensorImage inputTensorImg = TensorImage.fromFile('test.jpg');

  // inputTensorImg = imageProcessor.process(inputTensorImg);

  // // inference
  // interpreter.run(inputTensorImg, output);

  // // print the output
  // print(output);

  // print(interpreter.getInputTensors());
  // print(interpreter.getOutputTensors());

  // interpreter.close();
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
