## SuperGlue C++ Demo

### Building

First, generate [TorchScript](https://pytorch.org/tutorials/advanced/cpp_export.html) module files
of SuperPoint and SuperGlue by JIT-ing the annotated model definitions.

```bash
$ python3 ../jit.py
```

This should output `SuperPoint.zip` and `SuperGlue.zip`.

Building the demo project requires `libtorch` and OpenCV 3+. Follow the instructions in
[*Installing C++ Distributions of PyTorch*](https://pytorch.org/cppdocs/installing.html) for `libtorch` setup.

Create a build directory and configure CMake.

```bash
$ mkdir build && cd build
$ cmake .. -DCMAKE_PREFIX_PATH=<libtorch path>
$ make
```

### Usage

```.env
$ ./superglue <image0> <image1> <downscaled_width>
```

This will measure the average FPS over 50 iterations and outputs `matches.png` with a visualization of the detected keypoints and matches.