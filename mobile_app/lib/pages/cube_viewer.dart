import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_cube/flutter_cube.dart';

class CubeViewer extends StatelessWidget {
  final List<String> images;
  const CubeViewer({Key? key, required this.images}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('3D Cube Viewer')),
      body: Column(
        children: [
          // Show thumbnails of captured faces
          SizedBox(
            height: 100,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: images.length,
              itemBuilder: (context, idx) {
                return Padding(
                  padding: const EdgeInsets.all(4),
                  child: Image.file(
                    File(images[idx]),
                    width: 100,
                    fit: BoxFit.cover,
                  ),
                );
              },
            ),
          ),
          Expanded(
            child: Cube(
              onSceneCreated: (Scene scene) {
                scene.world.add(Object(
                  scale: Vector3.all(2.0),
                  fileName: 'assets/cube/cube.obj',
                ));
                scene.camera.zoom = 10;
              },
            ),
          ),
        ],
      ),
    );
  }
}