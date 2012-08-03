all:
	gcc -O2 -Wall -pedantic -shared  -fPIC vtkctm.c -static -lopenctm -Wl,-Bdynamic -o _vtkctm.so
