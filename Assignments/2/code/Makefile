all: p1 p2 p3

p1:	BasicDenoising.cpp
	g++ -o p1 BasicDenoising.cpp `pkg-config --cflags --libs opencv`

p2:	AnisotropicDiffusion.cpp
	g++ -o p2 AnisotropicDiffusion.cpp `pkg-config --cflags --libs opencv`	

p3: WeinerFilter.cpp
	g++ -o p3 WeinerFilter.cpp `pkg-config --cflags --libs opencv`	


clean:
	rm p1
	rm p2
	rm p3