import 'package:flutter/material.dart';
import 'pages/input_page.dart';
import 'pages/camera_page.dart';
import 'pages/cube_viewer.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Rubik Solver',
      initialRoute: '/',
      routes: {
        '/': (_) => const InputPage(),
        '/camera': (_) => const CameraPage(),
        '/viewer': (ctx) {
          final urls = ModalRoute.of(ctx)!.settings.arguments as Map<String, String>;
          return CubeViewer(textureUrls: urls);
        },
      },
      // safety net so unknown routes don’t crash:
      onUnknownRoute: (_) => MaterialPageRoute(builder: (_) => const InputPage()),
    );
  }
}

