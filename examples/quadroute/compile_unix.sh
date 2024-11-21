rm -f *.o, *.out, *.log, *.csv
gcc -pedantic -Wall -Wextra -g -O3 -c ../../src/pyquasoare/c_quasoare_utils.c
gcc -pedantic -Wall -Wextra -g -O3 -c ../../src/pyquasoare/c_quasoare_core.c 
gcc -pedantic -Wall -Wextra -g -O3 -c example_quadroute.c
gcc -pedantic -g -O3 *.o -lm -o example_quadroute.out 
./example_quadroute.out

