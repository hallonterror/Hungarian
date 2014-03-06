#pragma once

#include <Eigen\Dense>

using namespace Eigen;

template<class T, int SAMPLES, int DIMENSIONS> 
class Hungarian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	enum { ALL_OK, NOT_INITIALIZED };

public:
	Hungarian() : _Mat(NULL) { };
	~Hungarian() { if (_Mat) delete _Mat; };

	void CreateDistanceMap(const Matrix<T, SAMPLES, DIMENSIONS>& First, const Matrix<T, SAMPLES, DIMENSIONS>& Second)
	{
		if (_Mat == NULL)
			_Mat = new Array<T, SAMPLES, SAMPLES>();
		CreateDistanceMap(First, Second, *_Mat);
	}
	static void CreateDistanceMap(const Matrix<T, SAMPLES, DIMENSIONS>& First, const Matrix<T, SAMPLES, DIMENSIONS>& Second, Array<T, SAMPLES, SAMPLES>& Mat)
	{
		for (int x = 0; x < SAMPLES; x++)
			for (int y = 0; y < SAMPLES; y++)
			{
				Mat(y, x) = T();
				// Loop through combinations of data
				for (int i = 0; i < DIMENSIONS; i++)
				{
					// Add squared distances
					Mat(y, x) += (First(x, i) - Second(y, i))*(First(x, i) - Second(y, i));
				}
				// Take the square root to get euklidean distances
				Mat(y, x) = sqrt(Mat(y, x));
			}
	}
	int Apply(Matrix<int, SAMPLES, 2>& Idx)
	{
		if (_Mat != NULL)
			return Apply(*_Mat, Idx);
		return NOT_INITIALIZED;
	}
	int Apply(const Array<T, SAMPLES, SAMPLES>& Mat, Matrix<int, SAMPLES, 2>& Idx)
	{
		Idx = Matrix<int, SAMPLES, 2>::Zero();
		Array<T, SAMPLES, SAMPLES> tMat = Mat;

		// First remove min value from rows and then from cols
		for (int r = 0; r < SAMPLES; r++)
		{
			tMat.row(r) = tMat.row(r) - tMat.row(r).minCoeff();
		}
		for (int c = 0; c < SAMPLES; c++)
		{
			tMat.col(c) = tMat.col(c) - tMat.col(c).minCoeff();
		}

		// Attempt to make a selection

		// Count the number of zeros columnwise and rowwise
		Array<int, SAMPLES, SAMPLES> zMat = tMat.cwiseEqual(T()).cast<int>();

		// Select all lines to mark
		int addedLines = 0;
		while (zMat.sum() != 0)
		{
			Array<int, SAMPLES, 1> s1 = zMat.colwise().sum();
			Array<int, SAMPLES, 1> s2 = zMat.rowwise().sum();

			// Decide on the line to draw and set the corresponding values to zero
			Array<int, SAMPLES, 1>::Index l1idx, l2idx;
			if (s1.maxCoeff(&l1idx) < s2.maxCoeff(&l2idx))
				zMat.row(l2idx) = Array<int, 1, SAMPLES>::Zero();
			else
				zMat.col(l1idx) = Array<int, SAMPLES, 1>::Zero();

			addedLines++;
		}
		
		while (addedLines != SAMPLES)
		{
			// Implement solution to steps 6-8 http://www.wikihow.com/Use-the-Hungarian-Algorithm
			cout << "Stuck" << endl;
		}

		// Select such that there is one zero in each row
		// Step 9-10 http://www.wikihow.com/Use-the-Hungarian-Algorithm

		// Step 9 make optimal selection and set Idx
		// Step 10 compute the cost in Mat given this selection

		return ALL_OK;
	}

	int GetDistanceMap(Array<T, SAMPLES, SAMPLES>& Map)
	{
		if (_Mat)
		{
			Map = *_Mat;
			return ALL_OK;
		}
		return NOT_INITIALIZED;
	}

private:
	Array<T, SAMPLES, SAMPLES>* _Mat;
};

