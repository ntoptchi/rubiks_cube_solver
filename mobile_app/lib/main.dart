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
      },
      onGenerateRoute: (settings) {
        if (settings.name == '/viewer') {
          // Expecting a Map<String, String> of face->absoluteTextureUrl
          final urls = Map<String, String>.from(settings.arguments as Map);
          return MaterialPageRoute(
            builder: (_) => CubeViewer(textureUrls: urls),
          );
        }
        return null; // fallback to default
      },
    );
  }
}
