See CMakeLists.txt for dependencies.


To compile,
``` shell
mkdir build
cd build
cmake ..
```

To test, after compilation
``` shell
cd build
./test ../data/nuscenes_796b1988dd254f74bf2fb19ba9c5b8c6.ply
```

By default, PCL viewer should kick in and plot a beautiful ground segmentation and line fitting. 

It will also output `obstacle.ply` and `ground.ply` under `build/`. You can import them into meshlab for visualization. 
