import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const FoodCameraApp());
}

class FoodCameraApp extends StatelessWidget {
  const FoodCameraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Food Calorie Camera',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
        useMaterial3: true,
      ),
      home: const FoodCameraPage(),
    );
  }
}

class FoodCameraPage extends StatefulWidget {
  const FoodCameraPage({super.key});

  @override
  State<FoodCameraPage> createState() => _FoodCameraPageState();
}

class _FoodCameraPageState extends State<FoodCameraPage> {
  final ImagePicker _picker = ImagePicker();

  XFile? _imageFile;
  bool _isUploading = false;
  String? _statusMessage;

  // TODO: change this to your actual backend endpoint
  static const String _backendUrl =
      'https://your-backend-url.com/api/food/analyze';

  Future<void> _takePhoto() async {
    try {
      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        imageQuality: 85,
      );

      if (photo == null) return; // user cancelled

      setState(() {
        _imageFile = photo;
        _statusMessage = null;
      });
    } catch (e) {
      setState(() {
        _statusMessage = 'Error opening camera: $e';
      });
    }
  }

  Future<void> _uploadPhoto() async {
    if (_imageFile == null) {
      setState(() {
        _statusMessage = 'Take a picture first.';
      });
      return;
    }

    setState(() {
      _isUploading = true;
      _statusMessage = 'Uploading...';
    });

    try {
      final uri = Uri.parse(_backendUrl);

      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          await http.MultipartFile.fromPath(
            'image',              // field name your backend expects
            _imageFile!.path,
            filename: _imageFile!.name,
          ),
        );

      final response = await request.send();
      final body = await response.stream.bytesToString();

      if (!mounted) return;

      if (response.statusCode == 200) {
        // Later we’ll parse calories, macros, etc. from this JSON.
        setState(() {
          _statusMessage = 'Success! Server responded:\n$body';
        });
      } else {
        setState(() {
          _statusMessage =
              'Upload failed (${response.statusCode}):\n$body';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _statusMessage = 'Error uploading image: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isUploading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Food Calorie Camera'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            Expanded(
              child: Center(
                child: _imageFile == null
                    ? Text(
                        'Take a picture of your food to get started.',
                        style: theme.textTheme.bodyLarge,
                        textAlign: TextAlign.center,
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Image.file(
                          File(_imageFile!.path),
                          fit: BoxFit.cover,
                          width: double.infinity,
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 16),
            if (_statusMessage != null)
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  _statusMessage!,
                  style: theme.textTheme.bodySmall,
                ),
              ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isUploading ? null : _takePhoto,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Take Photo'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _isUploading ? null : _uploadPhoto,
                    icon: _isUploading
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.cloud_upload),
                    label: Text(_isUploading ? 'Uploading...' : 'Send'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
