# Sources and licenses

Google YAMNet v1: pretrained AudioSet model, Apache-2.0. Browser weights use the official Google TensorFlow.js export:
https://www.kaggle.com/models/google/yamnet/tfJs/tfjs/1
The export archive hash is recorded in models/catalog.json. Numerical parity with the Python SavedModel was checked on existing train/development inputs. License: [Apache-2.0](licenses/tensorflow-Apache-2.0.txt).

TensorFlow.js 4.22.0 is Apache-2.0. fft.js 4.0.4 is MIT. Bundled dependency notices are in app.js.LEGAL.txt. Model export/build dependencies are pinned by web/package-lock.json.

Playable real examples:

- Svanstrom/DroneDetectionThesis, drone, helicopter and background excerpts: CC0-1.0, https://github.com/DroneDetectionThesis/Drone-detection-dataset/tree/442d0c6750708cc221636f7a1fd31cbd8cb0aeb8 .
- Batear field drone and rural-background excerpts: MIT, https://github.com/batear-io/batear-datasets/tree/d0d6e67d9e7d167be9e0db249ef6a44acb52f732 . [Copyright and license](licenses/batear-MIT.txt).

Excerpts were converted to mono 16 kHz PCM16 for browser playback. Exact original URLs, hashes, offsets and publisher classes are in examples/catalog.json. Labels are weak recording labels; no independently verified audible event timing is implied.

D3 ground and imported insect audio were inspected locally. Their raw audio is not included in this site. The local research report records their source terms and provenance.

Trained experimental CNN/SVM/logistic heads retain the frozen project model identities, weights and decisions. No model training occurs on this site. Upload-selected audio and draft listening notes remain on the user's device.
