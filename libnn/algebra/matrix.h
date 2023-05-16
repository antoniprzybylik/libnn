#ifndef MATRIX_H_
#define MATRIX_H_

#include <ostream>
#include <cstdint>
#include <cstddef>

template<typename T>
class Matrix;

template<typename T>
class RowReference;

#include "vector.h"

template<typename T>
class Matrix {
	static_assert(std::is_arithmetic<T>::value, "");

private:
	size_t n, m;
	T **const tab;

public:
	Matrix(const size_t n, const size_t m);
	Matrix(const std::initializer_list<
			std::initializer_list<T> > c);
	Matrix(const Matrix&);
	~Matrix(void);

	size_t height(void) const;
	size_t width(void) const;

	Matrix<T> transpose(void) const;

	RowReference<T> operator[](const int i);
	const RowReference<T> operator[](const int i) const;

	Matrix &operator*=(const Matrix &M) &;
	Matrix &operator+=(const Matrix &M) &;
	Matrix &operator-=(const Matrix &M) &;

	template<typename U> friend
	std::ostream &operator<<(std::ostream&,
				 const Matrix<U>&);

	template<typename U> friend
	Matrix<U> operator*(Matrix<U> M1, const Matrix<U> &M2);
	template<typename U> friend
	Matrix<U> operator+(Matrix<U> M1, const Matrix<U> &M2);
	template<typename U> friend
	Matrix<U> operator-(Matrix<U> M1, const Matrix<U> &M2);

	template<typename U> friend
	Matrix<U> operator&(const Matrix<U> &M,
			    const ColumnVector<U> &v);
	template<typename U> friend
	Matrix<U> operator&(const ColumnVector<U> &v,
			    const Matrix<U> &M);

	template<typename U> friend
	ColumnVector<U> operator*(const Matrix<U> &M,
				  const ColumnVector<U> &v);
	template<typename U> friend
	RowVector<U> operator*(const RowVector<U> &v,
			       const Matrix<U> &M);
};

template<typename T>
class RowReference : public BasicVector<T> {
private:
	T *&tab_ref;

public:
	RowReference(const size_t len) = delete;
	RowReference(T *&tab, const size_t len) : BasicVector<T>(tab, len), tab_ref(tab) {};
	~RowReference(void) override {};

	friend class Vector<T>;

	template<typename U> friend
	void std::swap(RowReference<U>&&, RowReference<U>&&);

	RowReference &&operator+=(const RowVector<T> &v) &&;
	RowReference &&operator-=(const RowVector<T> &v) &&;
	RowReference &&operator+=(const RowReference<T> &&v) &&;
	RowReference &&operator-=(const RowReference<T> &&v) &&;
};

template<typename T>
void std::swap(RowReference<T> &&r1, RowReference<T> &&r2);

#endif /* MATRIX_H_ */
