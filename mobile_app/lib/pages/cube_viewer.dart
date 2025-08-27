import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter/material.dart';
import 'package:flutter_cube/flutter_cube.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

class CubeViewer extends StatefulWidget {
  /// keys: U,R,F,D,L,B -> absolute http(s) urls to the face textures
  final Map<String, String> textureUrls;
  const CubeViewer({Key? key, required this.textureUrls}) : super(key: key);

  @override
  State<CubeViewer> createState() => _CubeViewerState();
}

class _CubeViewerState extends State<CubeViewer> {
  String? _objPath; // local file path to the prepared obj in tmp

  @override
  void initState() {
    super.initState();
    _prepareBundle();
  }

  Future<void> _prepareBundle() async {
    // 1) Create temp dir: <tmp>/cube_bundle/textures
    final tmp = await getTemporaryDirectory();
    final bundleDir = Directory('${tmp.path}/cube_bundle');
    final texDir = Directory('${bundleDir.path}/textures');
    if (!await bundleDir.exists()) await bundleDir.create(recursive: true);
    if (!await texDir.exists()) await texDir.create(recursive: true);

    // 2) Download the 6 face textures to temp
    for (final f in const ['U', 'R', 'F', 'D', 'L', 'B']) {
      final url = widget.textureUrls[f]!;
      final resp = await http.get(Uri.parse(url));
      if (resp.statusCode != 200) {
        throw Exception('Failed to download $f.png from $url');
      }
      await File('${texDir.path}/$f.png').writeAsBytes(resp.bodyBytes);
    }

    // 3) Copy OBJ/MTL from assets -> temp (and rewrite map_Kd to our textures/)
    final objBytes = await rootBundle.load('assets/models/cube/cube.obj');
    final objPath = '${bundleDir.path}/cube.obj';
    await File(objPath).writeAsBytes(objBytes.buffer.asUint8List());

    final mtlSrc = await rootBundle.loadString('assets/models/cube/cube.mtl');
    final mtlPath = '${bundleDir.path}/cube.mtl';
    final mtlRewritten = _rewriteMtlMapKd(mtlSrc);
    await File(mtlPath).writeAsString(mtlRewritten);

    if (!mounted) return;
    setState(() => _objPath = objPath);
  }

  /// Ensure each face material maps to our downloaded textures/U.png, etc.
  String _rewriteMtlMapKd(String mtlText) {
    final faces = {'U','R','F','D','L','B'};
    final lines = mtlText.split('\n');
    final out = <String>[];
    String? current;

    for (final raw in lines) {
      final line = raw.trimRight();

      if (line.startsWith('newmtl ')) {
        current = line.substring(7).trim();
        out.add(line);
        continue;
      }

      if (line.startsWith('map_Kd ') && current != null && faces.contains(current)) {
        // Force texture path to textures/<face>.png relative to OBJ/MTL dir
        out.add('map_Kd textures/$current.png');
      } else {
        out.add(line);
      }
    }

    return out.join('\n');
  }

  @override
  Widget build(BuildContext context) {
    if (_objPath == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }
    return Scaffold(
      appBar: AppBar(title: const Text('3D Cube')),
      body: Cube(
        onSceneCreated: (scene) {
          scene.camera.zoom = 10;
          scene.world.add(Object(fileName: _objPath!));
        },
      ),
    );
  }
}
