all:
	gcc -O2 -Wall -pedantic -shared -fPIC vtkctm.c -lopenctm -o _vtkctm.so
