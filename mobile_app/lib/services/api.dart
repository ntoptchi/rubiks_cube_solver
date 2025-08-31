import 'dart:convert';
import 'package:http/http.dart' as http;

const baseUrl = 'http://127.0.0.1:8000/api'; // using adb reverse on a real device

Future<bool> pingHealth() async {
  try {
    final r = await http
        .get(Uri.parse(baseUrl.replaceFirst('/api', '/health')))
        .timeout(const Duration(seconds: 3));
    return r.statusCode == 200;
  } catch (_) {
    return false;
  }
}

Future<Map<String, dynamic>> uploadOneFace(String imagePath) async {
  final uri = Uri.parse('$baseUrl/scan_face');
  final req = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('image', imagePath));

  final resp = await req.send().timeout(const Duration(seconds: 15));
  final body = await resp.stream.bytesToString();
  if (resp.statusCode != 200) {
    throw Exception('scan_face failed: ${resp.statusCode} $body');
  }
  final map = jsonDecode(body) as Map<String, dynamic>;
  return map;
}

Future<Map<String, dynamic>> solveFromGrids(
  Map<String, List<List<String>>> grids,
) async {
  final uri = Uri.parse('$baseUrl/solve_from_grids');
  final faces = <String, dynamic>{};
  for (final f in ['U','R','F','D','L','B']) {
    faces[f] = {'grid': grids[f], 'rotation': 0};
  }
  final resp = await http
      .post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'faces': faces}),
      )
      .timeout(const Duration(seconds: 20));
  if (resp.statusCode != 200) {
    throw Exception(jsonDecode(resp.body)['detail'] ?? 'server error');
  }
  return jsonDecode(resp.body) as Map<String, dynamic>;
}
