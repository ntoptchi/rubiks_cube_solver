
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;


const String baseUrl = 'http://127.0.0.1:8000';

Future<bool> pingHealth() async {
  try {
    final uri = Uri.parse('$baseUrl/health');
    final r = await http.get(uri).timeout(const Duration(seconds: 3));
    debugPrint('Health: ${r.statusCode} ${r.body}');
    return r.statusCode == 200;
  } catch (e) {
    debugPrint('Health check failed: $e');
    return false;
  }
}
