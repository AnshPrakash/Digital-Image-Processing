all: p1 p3 ghdr p2

p1: LinearAndLogRescaling.cpp
	g++ -o p1 LinearAndLogRescaling.cpp `pkg-config --cflags --libs opencv`

p3: program3.cpp
	g++ -o p3 program3.cpp `pkg-config --cflags --libs opencv`

p2: DetailEnhancement.cpp
	g++ -o p2 DetailEnhancement.cpp `pkg-config --cflags --libs opencv`

ghdr: GradientHDR.cpp
	g++ -o ghdr GradientHDR.cpp `pkg-config --cflags --libs opencv`

clean:
	rm p1
	rm p2
	rm p3
	rm ghdr
