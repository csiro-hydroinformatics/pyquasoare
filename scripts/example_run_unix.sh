rm -f *.o, *.out, *.log, *.csv
gcc -pedantic -Wall -Wextra -g -O3 -c ../src/pyrezeq/c_rezeq_utils.c
gcc -pedantic -Wall -Wextra -g -O3 -c ../src/pyrezeq/c_rezeq_quad.c 
gcc -pedantic -Wall -Wextra -g -O3 -c example.c
gcc -pedantic -g -O3 *.o -lm -o example.out 
#./example.out

# To check memory leaks
valgrind -s --leak-check=yes ./example.out
