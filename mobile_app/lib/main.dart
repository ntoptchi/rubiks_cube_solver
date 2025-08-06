import 'package:flutter/material.dart';
import 'pages/input_page.dart';
import 'pages/camera_page.dart';
import 'pages/cube_viewer.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Rubik Solver',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const InputPage(),
      routes: {
        '/camera': (_) => const CameraPage(),
        '/viewer': (ctx) {
    final images = ModalRoute.of(ctx)!.settings.arguments as List<String>;
    return CubeViewer(images: images);
        },
      },
    );
  }
}