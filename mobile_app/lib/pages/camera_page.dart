import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  List<CameraDescription>? _cameras;
  CameraController? _controller;
  bool _isCameraReady = false;

  int _currentFace = 0;
  final List<String> _faceNames = ['up', 'right', 'front', 'down', 'left', 'back'];
  final List<String> _photos = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras!.first, ResolutionPreset.medium);
    await _controller!.initialize();
    setState(() {
      _isCameraReady = true;
    });
  }

  Future<void> _capture() async {
    if (!_isCameraReady || _controller == null) return;
    final file = await _controller!.takePicture();
    setState(() {
      _photos.add(file.path);
      _currentFace++;
    });
    if (_currentFace == 6) {
      _uploadAndSolve();
    }
  }

  Future<void> _uploadAndSolve() async {
    setState(() => _isLoading = true);

    final uri = Uri.parse('http://10.0.2.2:8000/scan');
    final request = http.MultipartRequest('POST', uri);
    for (var i = 0; i < 6; i++) {
      request.files.add(
        await http.MultipartFile.fromPath(
          _faceNames[i],
          _photos[i],
        ),
      );
    }

    final streamedResp = await request.send();
    final body = await streamedResp.stream.bytesToString();
    setState(() => _isLoading = false);

    if (streamedResp.statusCode == 200) {
      final data = jsonDecode(body) as Map<String, dynamic>;
      final moves = List<String>.from(data['solution']);

      const host = 'http://10.0.2.2:8000';
      const faces = ['U', 'R', 'F', 'D', 'L', 'B'];
      final textureUrls = faces
          .map((f) => '$host/static/textures/$f.png')
          .toList();

      Navigator.pushNamed(
        context,
        '/viewer',
        arguments: textureUrls,
      );
    } else {
      final error = (jsonDecode(body) as Map<String, dynamic>)['detail'] ?? body;
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Scan failed'),
          content: Text(error.toString()),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            )
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // Show loading during upload
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // Wait for camera initialization
    if (!_isCameraReady || _controller == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    // Capture each face in turn
    if (_currentFace < 6) {
      final faceLabel = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back'][_currentFace];
      return Scaffold(
        appBar: AppBar(title: Text('Capture $faceLabel Face')),
        body: CameraPreview(_controller!),
        floatingActionButton: FloatingActionButton(
          onPressed: _capture,
          child: const Icon(Icons.camera_alt),
        ),
      );
    }

    // Should never reach here because _uploadAndSolve() runs at _currentFace == 6
    return const Scaffold(body: SizedBox.shrink());
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
}
