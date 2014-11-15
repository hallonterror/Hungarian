
#include "Hungarian.h"
#include <iostream>

using namespace std;

enum TEST_CODES
{
	ALL_OK,
	INDEX1_DIFF,
	INDEX2_DIFF,
	ERROR_DIFF
};

enum DATA_SETS
{
	Set1_5x5,
	Set2_5x5,
	Set3_5x5,
	Set1_5x4
};

void getData(DATA_SETS i, Matrix<float, 5, 2>& f, Matrix<float, 5, 2>& s, Array<int, 5, 1>& tfs, Array<int, 5, 1>& tss, float& te)
{
	switch (i)
	{
	case Set1_5x5:
		f << 1, 2, 5, 7, 4, 6, 9, 8, 1, 4; // x1, y1, x2, y2, ...
		s << 8, 7, 4, 6, 2, 7, 8, 5, 1, 3;

		tfs << 4, 3, 1, 0, 2;
		tss << 3, 2, 4, 1, 0;
		te = 9.1820f;
		break;
	case Set2_5x5:
		f << 0.4517, -0.1303, 0.1837, -0.4762, 0.8620, -1.3617, 0.4550, -0.8487, -0.3349, 0.5528;
		s << 0.5152, 0.2614, -0.9415, -0.1623, -0.1461, -0.5320, 1.6821, -0.8757, -0.4838, -0.7120;

		tfs << 0, 2, 3, 4, 1;
		tss << 0, 4, 1, 2, 3;
		te = 3.5710f;
		break;
	case Set3_5x5:
		f << -1.1742, -0.1922, -0.2741, 1.5301, -0.2490, -1.0642, 1.6035, 1.2347, -0.2296, -1.5062;
		s << -0.4446, -0.1559, 0.2761, -0.2612, 0.4434, 0.3919, -1.2507, -0.9480, -0.7411, -0.5078;

		tfs << 4, 0, 1, 2, 3;
		tss << 1, 2, 3, 4, 0;
		te = 5.7876f;
		break;
	default:
		break;
	}
}
void getData(DATA_SETS i, Matrix<float, 5, 2>& f, Matrix<float, 4, 2>& s, Array<int, 5, 1>& tfs, Array<int, 4, 1>& tss, float& te)
{
	switch (i)
	{
	case Set1_5x4:
		f << -1.1742, -0.1922, -0.2741, 1.5301, -0.2490, -1.0642, 1.6035, 1.2347, -0.2296, -1.5062;
		s << -0.4446, -0.1559, 0.2761, -0.2612, 0.4434, 0.3919, -1.2507, -0.9480;

		tfs << 0, 2, 1, -1, 3;
		tss << 0, 2, 1, 4;
		te = 4.1991f;
	default:
		break;
	}
}

int testInstantiated5x5()
{
	// Create a Hungarian object with correct dimensions
	Hungarian<float, 2, 5> H;

	// Declare result variables
	float error = 0.0f;
	Array<int, 5, 1> Idx1, Idx2;

	// Declare ground truth variables
	float true_error;
	Array<int, 5, 1> true_Idx1, true_Idx2;

	// Create measurement data
	Matrix<float, 5, 2> first, second;

	for (int i = 0; i < 3; i++) // CHANGE TO 3 WHEN REMOVING CODE BELOW
	{
		DATA_SETS current = (DATA_SETS)i;

		// Perform first computation
		getData(current, first, second, true_Idx1, true_Idx2, true_error);
		H.CreateDistanceMap(first, second);
		H.Apply();
		error = H.GetError();
		H.GetFirstIndices(Idx1);
		H.GetSecondIndices(Idx2);

		if ((Idx1 - true_Idx1).abs().sum() > 0)
			return INDEX1_DIFF;
		if ((Idx2 - true_Idx2).abs().sum() > 0)
			return INDEX2_DIFF;
		if (std::abs(true_error - error) > 0.0001)
			return ERROR_DIFF;
	}

	return ALL_OK;
}

int testStatic5x4()
{
	int code = 0;

	// Declare result variables
	float error = 0.0f;
	Array<int, 5, 1> Idx1;
	Array<int, 4, 1> Idx2;

	// Declare ground truth variables
	float true_error;
	Array<int, 5, 1> true_Idx1;
	Array<int, 4, 1> true_Idx2;

	// Create measurement data
	Matrix<float, 5, 2> first;
	Matrix<float, 4, 2>	second;

	// Temporary objects
	Array<float, 5, 4> D;
	Array<int, 5, 4> S;

	// Perform computation
	getData(Set1_5x4, first, second, true_Idx1, true_Idx2, true_error);
	Hungarian<float, 2, 5, 4>::CreateDistanceMap(first, second, D);
	Hungarian<float, 2, 5, 4>::Apply(D, S);
	error = Hungarian<float, 2, 5, 4>::GetError(D, S);
	Hungarian<float, 2, 5, 4>::GetFirstIndices(S, Idx1);
	Hungarian<float, 2, 5, 4>::GetSecondIndices(S, Idx2);

	if ((Idx1 - true_Idx1).abs().sum() > 0)
		code += INDEX1_DIFF;
	if ((Idx2 - true_Idx2).abs().sum() > 0)
		code += INDEX2_DIFF;
	if (std::abs(true_error - error) > 0.0001)
		code += ERROR_DIFF;

	if (code != 0) code += 200;

	return code;
}

int main()
{
	cout << "Result from instantiated 5x5 tests: " << testInstantiated5x5() << endl;
	cout << "Result from static 5x4 test: " << testStatic5x4() << endl;
	return 1;
}