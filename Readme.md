__Workarounds:__
* If CUDA is installs, but pytorch fails with ```RuntimeError: nvrtc: error: failed to open nvrtc-builtins64_<required>.dll```, then it's probably a version mismatch. In this case,
  * Go to folder: ```NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin```
  * Duplicate ```nvrtc-builtins64_<installed>.dll```, and rename as  ```nvrtc-builtins64_<required>.dll```
  * Done