import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import '../utils/cube_model.dart';
import 'solution_page.dart';
import 'camera_page.dart';

class InputPage extends StatefulWidget {
  const InputPage({Key? key}) : super(key: key);

  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  final Map<String, List<String>> faces = {
    'U': List.filled(9, 'W'),
    'R': List.filled(9, 'R'),
    'F': List.filled(9, 'G'),
    'D': List.filled(9, 'Y'),
    'L': List.filled(9, 'O'),
    'B': List.filled(9, 'B'),
  };

  final List<String> colorCycle = ['W', 'R', 'G', 'Y', 'O', 'B'];
  final Map<String, Color> colorMap = {
    'W': Colors.white,
    'R': Colors.red,
    'G': Colors.green,
    'Y': Colors.yellow,
    'O': Colors.orange,
    'B': Colors.blue,
  };

  bool _isLoading = false;

  Future<List<String>> _solveCube() async {
    final faceStrings = faces.map((f, stickers) => MapEntry(f, stickers.join()));
    final cube = CubeState(faces: faceStrings);
    final url = Uri.parse('http://10.0.2.2:8000/solve');
    final body = jsonEncode(cube.toJson());
    print('Posting to \$url with body: \$body');
    final response = await http.post(url,
        headers: {'Content-Type': 'application/json'}, body: body);
    print('Status: \${response.statusCode}');
    print('Body: \${response.body}');
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return List<String>.from(data['solution']);
    } else {
      final err = jsonDecode(response.body);
      throw Exception(err['error'] ?? err['detail'] ?? 'Server error');
    }
  }

  void _onSolve() async {
    setState(() => _isLoading = true);
    try {
      final moves = await _solveCube();
      if (!mounted) return;
      Navigator.push(
          context,
          MaterialPageRoute(
              builder: (_) => SolutionPage(moves: moves),
          ),
      );
    } catch (e) {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Error'),
          content: Text(e.toString()),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Rubik Solver')),
      body: Center(
        child: ElevatedButton(
          onPressed: () => Navigator.pushNamed(context, '/camera'),
          child: const Text('Scan Cube with Camera'),
        ),
      ),
    );
  }
}