all:
	gcc -O2 -Wall -pedantic -shared  -fPIC vtkctm.c libopenctm.a -o _vtkctm.so
