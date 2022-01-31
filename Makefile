nbody: nbody.c
	pgcc -DDUMP nbody.c -o nbody -lm -ta:tesla -mp -Minfo=all -O3 

clean:
	rm -f nbody

