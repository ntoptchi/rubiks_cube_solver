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
  bool _isLoading = false;

  int _currentFace = 0;
  final List<String> _faceNames = ['up', 'right', 'front', 'down', 'left', 'back'];
  final List<String> _photos = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  void _showError(String msg) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Oops'),
        content: Text(msg),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          )
        ],
      ),
    );
  }

  Future<void> _uploadAndSolve() async {
    setState(() => _isLoading = true);

    try {
      final uri = Uri.parse('http://127.0.0.1:8000/scan');

      final request = http.MultipartRequest('POST', uri);
      for (var i = 0; i < 6; i++) {
        request.files.add(
          await http.MultipartFile.fromPath(_faceNames[i], _photos[i]),
        );
      }

      final streamedResp = await request
          .send()
          .timeout(const Duration(seconds: 15), onTimeout: () {
        throw Exception('Request timed out');
      });

      final body = await streamedResp.stream.bytesToString();
      print('Response [${streamedResp.statusCode}]: $body');
      setState(() => _isLoading = false);

      if (streamedResp.statusCode == 200) {
        final data = jsonDecode(body) as Map<String, dynamic>;
        final moves = List<String>.from(data['solution']);

        const host = 'http://192.168.1.49:8000';
        const faces = ['U', 'R', 'F', 'D', 'L', 'B'];
        final textureUrls = faces.map((f) => '$host/static/textures/$f.png').toList();

        Navigator.pushNamed(context, '/viewer', arguments: textureUrls);
      } else {
        final errorDetail = (jsonDecode(body) as Map<String, dynamic>)['detail'] ?? body;
        _showError('Scan failed: $errorDetail');
      }
    } catch (e) {
      setState(() => _isLoading = false);
      _showError('Upload error: $e');
    }
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras!.first, ResolutionPreset.medium);
    await _controller!.initialize();
    setState(() => _isCameraReady = true);
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

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (!_isCameraReady || _controller == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    if (_currentFace < 6) {
      final label = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back'][_currentFace];
      return Scaffold(
        appBar: AppBar(title: Text('Capture $label Face')),
        body: CameraPreview(_controller!),
        floatingActionButton: FloatingActionButton(
          onPressed: _capture,
          child: const Icon(Icons.camera_alt),
        ),
      );
    }
    return const Scaffold(body: SizedBox.shrink());
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
}
