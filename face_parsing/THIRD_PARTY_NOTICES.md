# Third-Party Notices

This library bundles or depends on the following third-party components. The
Apache License 2.0 in the repository root covers the library's own code. It does
not cover these components.

## face-parsing.PyTorch

- Source: https://github.com/zllrunning/face-parsing.PyTorch
- Pinned version: commit `d2e684cf1588b46145635e8fe7bcc29544e5537e`
- Included as: the git submodule `griptape_nodes_library_face_parsing/face_parsing_pytorch`
- License: MIT

The full license text from the upstream repository follows.

```text
MIT License

Copyright (c) 2019 zll

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Model weights

The face parsing weights (`79999_iter.pth`) are downloaded at runtime from the
HuggingFace repository `vivym/face-parsing-bisenet`. That repository is a
third-party mirror of the weights that the upstream project distributes through
Google Drive. The upstream README does not state a separate license for the
weights, and the HuggingFace repository does not declare one. Check that your use
of the weights is permitted before you rely on them.

## Python dependencies

The runtime dependencies (`torch`, `torchvision`, `numpy`, `pillow`, and
`huggingface_hub`) are installed from PyPI under their own licenses. They are not
redistributed by this repository.
