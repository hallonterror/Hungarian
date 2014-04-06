#pragma once

#include <Eigen\Dense>
#include <vector>

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
		// Implemented according to: http://www.wikihow.com/Use-the-Hungarian-Algorithm

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

		Array<bool, SAMPLES, 1> coveredCols, coveredRows;
		int addedLines = CoverWithLines(tMat, coveredCols, coveredRows);

		while (addedLines != SAMPLES)
		{
			// Step 6, find minimum value that is not covered
			T minValue = numeric_limits<T>::max();
			for (int c = 0; c < SAMPLES; c++)
				for (int r = 0; r < SAMPLES; r++)
					if (coveredRows(r) != 1 && coveredCols(c) != 1)
						minValue = std::min<T>(minValue, tMat(r, c));

			// Add the value to the covered elements
			for (int c = 0; c < SAMPLES; c++)
				if (coveredCols(c))
					tMat.col(c) = tMat.col(c) + minValue;
			for (int r = 0; r < SAMPLES; r++) // Two times if covered twice
				if (coveredRows(r))
					tMat.row(r) = tMat.row(r) + minValue;

			// Step 7, subtract the minimum element
			tMat = tMat - minValue;

			// Step 8, Try to cover again
			addedLines = CoverWithLines(tMat, coveredCols, coveredRows);
		}

		// Step 9, select zero elements
		SelectZeros(tMat, Idx);

		return ALL_OK;
	}
	T GetError(const Array<T, SAMPLES, SAMPLES>& Mat, const Matrix<int, SAMPLES, 2>& Idx)
	{
		T error = T();
		for (int r = 0; r < SAMPLES; r++)
			error += Mat(Idx(r, 0), Idx(r, 1));

		return error;
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

	int CoverWithLines(const Array<T, SAMPLES, SAMPLES>& tMat, Array<bool, SAMPLES, 1>& coveredCols, Array<bool, SAMPLES, 1>& coveredRows)
	{
		// Create a matrix with ones where there are zeros in the distance map
		Array<int, SAMPLES, SAMPLES> zMat = tMat.cwiseEqual(T()).cast<int>();

		// Prepare results
		coveredCols = Array<bool, SAMPLES, 1>::Zero();
		coveredRows = Array<bool, SAMPLES, 1>::Zero();
		int addedLines = 0;

		// Loop until all covered
		while (zMat.sum() != 0) // All zeros are marked
		{
			// Create vector with the number of zeros along rows and columns
			Array<int, SAMPLES, 1> colZeros = zMat.colwise().sum();
			Array<int, SAMPLES, 1> rowZeros = zMat.rowwise().sum();

			// Decide on the line to draw and set the corresponding values to zero
			Array<int, SAMPLES, 1>::Index colIdx, rowIdx;
			if (colZeros.maxCoeff(&colIdx) < rowZeros.maxCoeff(&rowIdx))
			{
				zMat.row(rowIdx) = Array<int, 1, SAMPLES>::Zero();
				coveredRows(rowIdx) = 1;
			}
			else
			{
				zMat.col(colIdx) = Array<int, SAMPLES, 1>::Zero();
				coveredCols(colIdx) = 1;
			}
			addedLines++;
		}

		return addedLines;
	}
	int SelectZeros(const Array<T, SAMPLES, SAMPLES>& tMat, Matrix<int, SAMPLES, 2>& Idx)
	{	// Step 9 make optimal selection and set Idx
		Array<int, SAMPLES, SAMPLES> zMat = tMat.cwiseEqual(T()).cast<int>();

		int addedIndices = 0;
		int iteration = 0;
		Array<int, SAMPLES, SAMPLES>::Index i;

		while (iteration < 10)
		{
			Array<int, SAMPLES, 1> colZeros = zMat.colwise().sum();
			for (int c = 0; c < SAMPLES; c++)
			{
				if (colZeros(c) == 1)
				{
					zMat.col(c).maxCoeff(&i);
					Idx(addedIndices, 0) = i;
					Idx(addedIndices, 1) = c;
					zMat.row(i) = Array<int, 1, SAMPLES>::Zero();
					addedIndices++;
				}

				if (addedIndices == SAMPLES)
					break;
			}


			Array<int, SAMPLES, 1> rowZeros = zMat.rowwise().sum();
			for (int r = 0; r < SAMPLES; r++)
			{
				if (rowZeros(r) == 1)
				{
					zMat.row(r).maxCoeff(&i);
					Idx(addedIndices, 0) = r;
					Idx(addedIndices, 1) = i;
					zMat.col(i) = Array<int, SAMPLES, 1>::Zero();
					addedIndices++;
				}

				if (addedIndices == SAMPLES)
					break;
			}

			// Make sure that we dont loop forever (should not happen)
			iteration++;
		}

		return iteration == 10; // if iteration roof was hit, don't return 0
	}
};

