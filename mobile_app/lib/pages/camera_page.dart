import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../services/api.dart'; // uploadOneFace, solveFromGrids, baseUrl

class CameraPage extends StatefulWidget {
  const CameraPage({Key? key}) : super(key: key);

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  List<CameraDescription>? _cameras;
  CameraController? _controller;

  bool _isCameraReady = false;
  bool _isUploading = false;

  // New per-face flow state
  final List<String> _faceOrder = ['U', 'R', 'F', 'D', 'L', 'B'];
  int _currentFace = 0;
  final Map<String, List<List<String>>> _grids = {};

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras!.first, ResolutionPreset.medium);
    await _controller!.initialize();
    if (mounted) setState(() => _isCameraReady = true);
  }

  void _showError(String msg) {
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Oops'),
        content: Text(msg),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK')),
        ],
      ),
    );
  }

  Future<void> _capture() async {
    if (!_isCameraReady || _controller == null || _isUploading) return;

    final shot = await _controller!.takePicture();
    final capturedPath = shot.path;

    final String currentLabel = _faceOrder[_currentFace];

    setState(() => _isUploading = true);
    try {
      // Upload ONE face → get 3×3 grid
      final res = await uploadOneFace(capturedPath);
      final grid = (res['grid'] as List)
          .map((row) => List<String>.from(row as List))
          .toList();
      _grids[currentLabel] = grid;

      // Next face
      setState(() => _currentFace++);

      // When all six faces captured → solve
      if (_currentFace == _faceOrder.length) {
        final solved = await solveFromGrids(_grids);
        final textures = Map<String, dynamic>.from(solved['textures'] as Map);

        // Build absolute URLs for viewer
        final images = ['U','R','F','D','L','B']
            .map((f) => '$baseUrl${textures[f]}')
            .toList();

        if (!mounted) return;
        Navigator.pushNamed(context, '/viewer', arguments: images);
        // Or navigate to your moves page using solved['solution'] if you prefer
      }
    } catch (e) {
      if (!mounted) return;
      _showError('Scan failed: $e');
    } finally {
      if (mounted) setState(() => _isUploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraReady || _controller == null) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final faceLabel = _faceOrder[_currentFace];
    return Scaffold(
      appBar: AppBar(title: Text('Capture $faceLabel face')),
      body: Stack(
        children: [
          CameraPreview(_controller!),
          if (_isUploading)
            Container(
              color: Colors.black45,
              child: const Center(child: CircularProgressIndicator()),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _isUploading ? null : _capture,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }
}
