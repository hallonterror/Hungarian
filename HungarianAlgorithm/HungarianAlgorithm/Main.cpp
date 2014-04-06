
#include "Hungarian.h"
#include <iostream>

using namespace std;

int main()
{
	Hungarian<float, 5, 2> H;
	Matrix<float, 5, 2> first;
	Matrix<float, 5, 2> second;


	first << 1, 2, 5, 7, 4, 6, 9, 8, 1, 4;
	second << 8, 7, 4, 6, 2, 7, 8, 5, 1, 3;

	cout << first << endl << endl << second << endl;

	Array<float, 5, 5> M1, M2;

	H.CreateDistanceMap(first, second);
	H.GetDistanceMap(M1);

	Hungarian<float, 5, 2>::CreateDistanceMap(first, second, M2);

	//cout << endl << M1 << endl << endl << M2 << endl << endl;

	Matrix<int, 5, 2> Idx;
	H.Apply(Idx);

	cout << endl << Idx << endl;

	cout << "Error: " << H.GetError(M1, Idx) << endl;

	return 1;
}