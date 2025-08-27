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

  // Torch toggle
  bool _torchOn = false;

  // Per-face flow state
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
    // make sure we’re not using flash by default
    try {
      await _controller!.setFlashMode(FlashMode.off);
    } catch (_) {}
    if (mounted) setState(() => _isCameraReady = true);
  }

  Future<void> _setTorch(bool on) async {
    if (_controller == null) return;
    try {
      await _controller!.setFlashMode(on ? FlashMode.torch : FlashMode.off);
      if (mounted) setState(() => _torchOn = on);
    } catch (e) {
      // Device might not support torch; ignore but log.
      debugPrint('Torch not available: $e');
    }
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

    try {
      setState(() => _isUploading = true);

      // CHANGED: Respect the toggle; do NOT force the flash.
      try {
        await _controller!.setFlashMode(_torchOn ? FlashMode.torch : FlashMode.off);
      } catch (_) {}

      final XFile shot = await _controller!.takePicture();
      final capturedPath = shot.path;

      final String currentLabel = _faceOrder[_currentFace];

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
      // after you get `solved` from /solve_from_grids
      final textures = Map<String, dynamic>.from(solved['textures'] as Map);

      // Build face → absolute URL map
      final textureMap = <String, String>{
        for (final f in ['U','R','F','D','L','B']) f: '$baseUrl${textures[f] as String}',
      };

      // Navigate passing the Map
      Navigator.pushNamed(context, '/viewer', arguments: textureMap);

        // Optional: reset state so coming back lets you rescan immediately
        // setState(() {
        //   _currentFace = 0;
        //   _grids.clear();
        // });
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

    // After the last face is captured, avoid indexing _faceOrder[_currentFace]
    if (_currentFace >= _faceOrder.length) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final faceLabel = _faceOrder[_currentFace];
    return Scaffold(
      appBar: AppBar(
        title: Text('Capture $faceLabel face'),
        actions: [
          IconButton(
            tooltip: _torchOn ? 'Turn torch off' : 'Turn torch on',
            icon: Icon(_torchOn ? Icons.flash_on : Icons.flash_off),
            onPressed: () => _setTorch(!_torchOn),
          ),
        ],
      ),
      body: Stack(
        children: [
          CameraPreview(_controller!),

          // Overlay box to help framing the cube face (center square)
          IgnorePointer(
            child: Center(
              child: AspectRatio(
                aspectRatio: 1,
                child: Container(
                  margin: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.white70, width: 3),
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ),
          ),

          // Loading veil during upload
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
    // Make sure torch is off when leaving
    _controller?.setFlashMode(FlashMode.off);
    _controller?.dispose();
    super.dispose();
  }
}


