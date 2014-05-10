#pragma once

#include <Eigen\Dense>
#include <vector>

using namespace Eigen;

template<class T, int DIMENSIONS, int SAMPLES, int SAMPLES2 = SAMPLES>
class Hungarian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	enum { ALL_OK, NOT_INITIALIZED };

public:
	Hungarian() : _Mat(NULL) { };
	~Hungarian() { if (_Mat) delete _Mat; };

	// Static functionality (main work is done here)
	static void CreateDistanceMap(const Matrix<T, SAMPLES, DIMENSIONS>& First, const Matrix<T, SAMPLES2, DIMENSIONS>& Second, Array<T, SAMPLES, SAMPLES2>& Mat)
	{
		for (int x = 0; x < SAMPLES2; x++)
			for (int y = 0; y < SAMPLES; y++)
			{
				Mat(y, x) = T();
				// Loop through combinations of data
				for (int i = 0; i < DIMENSIONS; i++)
				{
					// Add squared distances
					Mat(y, x) += (First(y, i) - Second(x, i))*(First(y, i) - Second(x, i));
				}
				// Take the square root to get euklidean distances
				Mat(y, x) = sqrt(Mat(y, x));
			}
	}
	static int Apply(const Array<T, SAMPLES, SAMPLES2>& Mat, Array<int, SAMPLES, SAMPLES2>& Selection)
	{
		// Implemented according to: http://www.wikihow.com/Use-the-Hungarian-Algorithm

		const int maxSamples = std::max(SAMPLES, SAMPLES2);
		Array<T, Dynamic, Dynamic> tMat = Array<T, Dynamic, Dynamic>::Ones(maxSamples, maxSamples);
		tMat *= Mat.maxCoeff();
		tMat.block<SAMPLES, SAMPLES2>(0, 0) = Mat;

		// First remove min value from rows and then from cols
		for (int r = 0; r < maxSamples; r++)
		{
			tMat.row(r) = tMat.row(r) - tMat.row(r).minCoeff();
		}
		for (int c = 0; c < maxSamples; c++)
		{
			tMat.col(c) = tMat.col(c) - tMat.col(c).minCoeff();
		}

		Array<bool, Dynamic, 1> coveredCols = Array<bool, Dynamic, 1>(maxSamples);
		Array<bool, Dynamic, 1> coveredRows = Array<bool, Dynamic, 1>(maxSamples);
		int addedLines = CoverWithLines(tMat, coveredCols, coveredRows);

		while (addedLines != maxSamples)
		{
			// Step 6, find minimum value that is not covered
			T minValue = numeric_limits<T>::max();
			for (int c = 0; c < maxSamples; c++)
				for (int r = 0; r < maxSamples; r++)
					if (coveredRows(r) != 1 && coveredCols(c) != 1)
						minValue = std::min<T>(minValue, tMat(r, c));

			// Add the value to the covered elements
			for (int c = 0; c < maxSamples; c++)
				if (coveredCols(c))
					tMat.col(c) = tMat.col(c) + minValue;
			for (int r = 0; r < maxSamples; r++) // Two times if covered twice
				if (coveredRows(r))
					tMat.row(r) = tMat.row(r) + minValue;

			// Step 7, subtract the minimum element
			tMat = tMat - minValue;

			// Step 8, Try to cover again
			addedLines = CoverWithLines(tMat, coveredCols, coveredRows);
		}

		// Step 9, select zero elements
		return SelectZeros(tMat, Selection);
	}
	static T GetError(const Array<T, SAMPLES, SAMPLES2>& Mat, const Array<int, SAMPLES, SAMPLES2>& Selection)
	{
		T error = T();
		for (int x = 0; x < SAMPLES2; x++)
			for (int y = 0; y < SAMPLES; y++)
				if (Selection(y, x) == 1)
					error += Mat(y, x);
		return error;
	}
	static void GetFirstIndices(const Array<int, SAMPLES, SAMPLES2>& Selection, Array<int, SAMPLES, 1>& Idx)
	{
		Idx = -Array<int, SAMPLES, 1>::Ones();
		for (int r = 0; r < SAMPLES; r++)
			for (int c = 0; c < SAMPLES2; c++)
				if (Selection(r, c) == 1)
				{
					Idx(r) = c;
					break;
				}
	}
	static void GetSecondIndices(const Array<int, SAMPLES, SAMPLES2>& Selection, Array<int, SAMPLES2, 1>& Idx)
	{
		Idx = -Array<int, SAMPLES2, 1>::Ones();
		for (int c = 0; c < SAMPLES2; c++)
			for (int r = 0; r < SAMPLES; r++)
				if (Selection(r, c) == 1)
				{
					Idx(c) = r;
					break;
				}
	}

	// For instantiated object
	void CreateDistanceMap(const Matrix<T, SAMPLES, DIMENSIONS>& First, const Matrix<T, SAMPLES2, DIMENSIONS>& Second)
	{
		if (_Mat == NULL)
			_Mat = new Array<T, SAMPLES, SAMPLES2>();
		CreateDistanceMap(First, Second, *_Mat);
	}
	int Apply()
	{
		if (_Mat != NULL)
		{
			if (_Sel == NULL)
				_Sel = new Array<int, SAMPLES, SAMPLES2>();
			return Apply(*_Mat, *_Sel);
		}
		return NOT_INITIALIZED;
	}

	void SetDistanceMap(const Array<T, SAMPLES, SAMPLES2>& Map)
	{
		*_Mat = Map;
	}
	int GetDistanceMap(Array<T, SAMPLES, SAMPLES2>& Map)
	{
		if (_Mat)
		{
			Map = *_Mat;
			return ALL_OK;
		}
		return NOT_INITIALIZED;
	}

	void SetSelection(const Array<int, SAMPLES, SAMPLES2>& Selection)
	{
		*_Idx = Idx;
	}
	int GetSelection(Array<int, SAMPLES, SAMPLES2>& Selection)
	{
		if (_Sel)
		{
			Selection = *_Sel;
			return ALL_OK;
		}
		return NOT_INITIALIZED;
	}

	T GetError()
	{
		if (_Mat && _Sel)
			return GetError(*_Mat, *_Sel);
		return NOT_INITIALIZED;
	}
	int GetFirstIndices(Array<int, SAMPLES, 1>& Idx)
	{
		if (_Mat == NULL || _Sel == NULL)
			return NOT_INITIALIZED;
		GetFirstIndices(*_Sel, Idx);

	}
	int GetSecondIndices(Array<int, SAMPLES2, 1>& Idx)
	{
		if (_Mat == NULL || _Sel == NULL)
			return NOT_INITIALIZED;
		GetSecondIndices(*_Sel, Idx);
	}

private:
	Array<T, SAMPLES, SAMPLES2>* _Mat;
	Array<int, SAMPLES, SAMPLES2>* _Sel;

	static int CoverWithLines(const Array<T, Dynamic, Dynamic>& tMat, Array<bool, Dynamic, 1>& coveredCols, Array<bool, Dynamic, 1>& coveredRows)
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
	static int SelectZeros(const Array<T, Dynamic, Dynamic>& Mat, Array<int, SAMPLES, SAMPLES2>& Sel)
	{	// Step 9 make optimal selection and set Idx
		const int maxSamples = std::max(SAMPLES, SAMPLES2);
		Array<int, Dynamic, Dynamic> zMat = Mat.cwiseEqual(T()).cast<int>();
		Array<int, Dynamic, Dynamic> tIdx = Array<int, Dynamic, Dynamic>::Zero(maxSamples, maxSamples);

		int addedIndices = 0;
		int iteration = 0;
		Array<int, Dynamic, Dynamic>::Index i;

		while (iteration < 10)
		{
			Array<int, Dynamic, 1> colZeros = zMat.colwise().sum();
			for (int c = 0; c < maxSamples; c++)
			{
				if (colZeros(c) == 1)
				{
					zMat.col(c).maxCoeff(&i);
					tIdx(i, c) = 1;
					zMat.row(i) = Array<int, 1, Dynamic>::Zero(maxSamples);
					addedIndices++;
				}

				if (addedIndices == SAMPLES2)
					break;
			}


			Array<int, Dynamic, 1> rowZeros = zMat.rowwise().sum();
			for (int r = 0; r < maxSamples; r++)
			{
				if (rowZeros(r) == 1)
				{
					zMat.row(r).maxCoeff(&i);
					tIdx(r, i) = 1;
					zMat.col(i) = Array<int, Dynamic, 1>::Zero(maxSamples);
					addedIndices++;
				}

				if (addedIndices == SAMPLES)
					break;
			}

			// Make sure that we dont loop forever (should not happen)
			iteration++;
		}

		Sel = tIdx.topLeftCorner(SAMPLES, SAMPLES2);
		return iteration == 10; // if iteration roof was hit, don't return 0
	}
};

