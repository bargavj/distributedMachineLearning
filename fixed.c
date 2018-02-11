#include "fixed.h"

fixed_t double_to_fixed(double d, int p) {
	return (fixed_t) (d * (1LL << p));
}

double fixed_to_double(fixed_t f, int p) {
	return (double)(((long double) f) / (1LL << p));
}

