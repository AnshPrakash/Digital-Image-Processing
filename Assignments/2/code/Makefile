all: p1 p2 p3 p4def p4mo

p1:	BasicDenoising.cpp
	g++ -o p1 BasicDenoising.cpp `pkg-config --cflags --libs opencv`

p2:	AnisotropicDiffusion.cpp
	g++ -o p2 AnisotropicDiffusion.cpp `pkg-config --cflags --libs opencv`	

p3: WeinerFilter.cpp
	g++ -o p3 WeinerFilter.cpp `pkg-config --cflags --libs opencv`	

p4def: PSFdefocus.cpp
	g++ -o p4def PSFdefocus.cpp `pkg-config --cflags --libs opencv`		

p4mo: PSFMotion.cpp
	g++ -o p4mo PSFMotion.cpp `pkg-config --cflags --libs opencv`		


clean:
	rm p1
	rm p2
	rm p3
	rm p4def
	rm p4mo