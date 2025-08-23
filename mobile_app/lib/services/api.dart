import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart';

/// Use 127.0.0.1 with `adb reverse` on a real device.
/// On the Android emulator, use http://10.0.2.2:8000 instead.
const String baseUrl = 'http://127.0.0.1:8000';

Future<Map<String, dynamic>> uploadOneFace(String imagePath) async {
  final uri = Uri.parse('$baseUrl/scan_face');
  final req = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('image', imagePath));
  debugPrint('→ POST $uri (image=$imagePath)');
  final streamed = await req.send();
  final body = await streamed.stream.bytesToString();
  debugPrint('← ${streamed.statusCode} $body');
  if (streamed.statusCode != 200) {
    throw Exception('scan_face failed: ${streamed.statusCode} $body');
  }
  return jsonDecode(body) as Map<String, dynamic>;
}

Future<Map<String, dynamic>> solveFromGrids(
  Map<String, List<List<String>>> grids,
) async {
  // Backend expects: {"faces": { "U": {"grid":[...], "rotation":0}, ... }}
  const order = ['U', 'R', 'F', 'D', 'L', 'B'];
  final faces = <String, dynamic>{};
  for (final f in order) {
    final g = grids[f];
    if (g == null) throw ArgumentError('Missing face $f');
    faces[f] = {
      'grid': g,
      'rotation': 0, // we let the backend auto-rotate
    };
  }

  final uri = Uri.parse('$baseUrl/solve_from_grids');
  final body = jsonEncode({'faces': faces});
  debugPrint('→ POST $uri');
  final resp = await http.post(
    uri,
    headers: {'Content-Type': 'application/json'},
    body: body,
  );
  debugPrint('← ${resp.statusCode} ${resp.body}');
  if (resp.statusCode != 200) {
    throw Exception('solve_from_grids failed: ${resp.statusCode} ${resp.body}');
  }
  return jsonDecode(resp.body) as Map<String, dynamic>;
}

Future<bool> pingHealth() async {
  try {
    final r = await http
        .get(Uri.parse('$baseUrl/health'))
        .timeout(const Duration(seconds: 3));
    return r.statusCode == 200;
  } catch (_) {
    return false;
  }
}

