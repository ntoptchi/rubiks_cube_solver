import 'package:flutter/material.dart';
import '../services/api.dart'; // pingHealth()

class InputPage extends StatefulWidget {
  const InputPage({Key? key}) : super(key: key);

  @override
  State<InputPage> createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  @override
  void initState() {
    super.initState();
    // Ping backend once the first frame is ready
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final ok = await pingHealth();
      if (!ok && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'Backend unreachable. If on a real device, run: adb reverse tcp:8000 tcp:8000',
            ),
            duration: Duration(seconds: 5),
          ),
        );
      }
    });
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
