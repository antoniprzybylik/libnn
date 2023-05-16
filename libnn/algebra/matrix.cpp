#include <memory>
#include <algorithm>
#include <initializer_list>

#include "matrix.h"

/* Definicje szablonów. */

template<typename T>
Matrix<T>::Matrix(const std::initializer_list<
				std::initializer_list<T> > c) :
tab(new T*[c.size()])
{
	typename std::initializer_list<
			std::initializer_list<T> >::const_iterator it;
	size_t i;
	size_t n, m;

	n = c.size();
	if (n < 1) {
		throw std::runtime_error(
			"Matrix dimensions have "
			"to be positive integers.");
	}

	it = c.begin();
	m = (*it).size();
	for ( ; it != c.end(); it++) {
		if ((*it).size() != m) {
			throw std::runtime_error(
				"Initializer list "
				"rows are of different "
				"lengths.");
		}
	}

	if (m < 1) {
		throw std::runtime_error(
			"Matrix dimensions have "
			"to be positive integers.");
	}

	for (it = c.begin(), i = 0;
	     it != c.end();
	     it++, i++) {
		tab[i] = new T[m];
		std::copy((*it).begin(),
			  (*it).end(),
			  this->tab[i]);
	}

	this->n = n;
	this-> m = m;
}

template<typename T>
Matrix<T>::Matrix(size_t n, size_t m) :
n(n),
m(m),
tab(new T*[n])
{
	size_t i;

	if (n < 1 || m < 1) {
		throw std::runtime_error(
			"Matrix dimensions have "
			"to be positive integers.");
	}

	for (i = 0; i < n; i++)
		tab[i] = new T[m];
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &M) :
n(M.n),
m(M.m),
tab(new T*[n])
{
	size_t i, j;

	for (i = 0; i < n; i++) {
		tab[i] = new T[m];
		for (j = 0; j < m; j++)
			tab[i][j] = M[i][j];
	}
}

template<typename T>
Matrix<T>::~Matrix(void)
{
	size_t i;

	for (i = 0; i < n; i++)
		delete[] tab[i];

	delete[] tab;
}

template<typename T>
size_t Matrix<T>::height(void) const
{
	return this->n;
}

template<typename T>
size_t Matrix<T>::width(void) const
{
	return this->m;
}

template<typename T>
Matrix<T> Matrix<T>::transpose(void) const
{
	Matrix<T> M(m, n);
	size_t i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++)
			M[i][j] = tab[j][i];
	}

	return M;
}

template<typename T>
RowReference<T> Matrix<T>::operator[](int i)
{
	return RowReference(tab[i], m);
}

template<typename T>
const RowReference<T> Matrix<T>::operator[](int i) const
{
	return RowReference(tab[i], m);
}

template<typename T>
Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &M) &
{
	T sum;
	size_t i, j, k;

	if (this->m != M.n) {
		throw std::runtime_error(
			"Wrong matrix dimensions "
			"for multiplication.");
	}

	Matrix<T> C = *this;

	this->m = M.m;
	for (i = 0; i < this->n; i++) {
		delete[] tab[i];
		tab[i] = new T[this->m];
	}

	for (i = 0; i < M.m; i++) {
		for (j = 0; j < C.n; j++) {
			sum = 0;
			for (k = 0; k < C.m; k++)
				sum += C[j][k]*M.tab[k][i];
			this->tab[j][i] = sum;
		}
	}
	
	return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &M) &
{
	size_t i, j;

	if (this->n != M.n ||
	    this->m != M.m) {
		throw std::runtime_error(
			"Wrong matrix dimensions "
			"for addition.");
	}

	for (i = 0; i < this->n; i++) {
		for (j = 0; j < this->m; j++)
			tab[i][j] += M.tab[i][j];
	}
	
	return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &M) &
{
	size_t i, j;

	if (this->n != M.n ||
	    this->m != M.m) {
		throw std::runtime_error(
			"Wrong matrix dimensions "
			"for substraction.");
	}

	for (i = 0; i < this->n; i++) {
		for (j = 0; j < this->m; j++)
			tab[i][j] -= M.tab[i][j];
	}
	
	return *this;
}

static inline
std::string alignr(const std::string &s, size_t len)
{
	return (std::string(len - s.size(), ' ')+s);
}

template<typename T>
std::ostream &operator<<(std::ostream &os,
			 const Matrix<T> &M)
{
	size_t i, j;
	size_t n = M.n;
	size_t m = M.m;

	std::unique_ptr<
		std::unique_ptr<std::string[]>[]> A;
	size_t max_len = 0;

	A = std::unique_ptr<std::unique_ptr<std::string[]>[]>(
			new std::unique_ptr<std::string[]>[n]);
	for (i = 0; i < n; i++) {
		A[i] = std::make_unique<std::string[]>(m);
		for (j = 0; j < m; j++) {
			A[i][j] = std::to_string(M[i][j]);
			max_len = std::max(max_len, A[i][j].size());
		}
	}

	os << "[";
	for (i = 0; i < n; i++) {
		if (i !=0)
			os << ' ';

		os << '[';
		for (j = 0; j < m-1; j++)
			os << alignr(A[i][j], max_len) << ", ";
		os << alignr(A[i][m-1], max_len) << ']';

		if (i != n-1)
			os << "\n";
	}
	os << ']';

	return os;
}

template <typename T>
Matrix<T> operator*(Matrix<T> M1, const Matrix<T> &M2)
{
	return (M1 *= M2);
}

template <typename T>
Matrix<T> operator+(Matrix<T> M1, const Matrix<T> &M2)
{
	return (M1 += M2);
}

template <typename T>
Matrix<T> operator-(Matrix<T> M1, const Matrix<T> &M2)
{
	return (M1 -= M2);
}

template<typename T>
void std::swap(RowReference<T> &&r1, RowReference<T> &&r2)
{
	T *c = r1.tab_ref;
	r1.tab_ref = r2.tab_ref;
	r2.tab_ref = c;
}

template<typename T>
RowReference<T> &&RowReference<T>::operator+=(const RowVector<T> &v) &&
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] += v.tab[i];

	return std::move(*this);
}

template<typename T>
RowReference<T> &&RowReference<T>::operator-=(const RowVector<T> &v) &&
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] -= v.tab[i];

	return std::move(*this);
}

template<typename T>
RowReference<T> &&RowReference<T>::operator+=(const RowReference<T> &&v) &&
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] += v.tab[i];

	return std::move(*this);
}

template<typename T>
RowReference<T> &&RowReference<T>::operator-=(const RowReference<T> &&v) &&
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] -= v.tab[i];

	return std::move(*this);
}

template<typename T>
Matrix<T> operator&(const Matrix<T> &M,
		    const ColumnVector<T> &v)
{
	size_t i, j;

	if (M.n != v.len) {
		throw std::runtime_error(
			"Wrong lengths "
			"for concatenation.");
	}

	Matrix<T> B(M.n, M.m+1);

	for (i = 0; i < M.n; i++) {
		for (j = 0; j < M.m; j++)
			B.tab[i][j] = M.tab[i][j];
		B.tab[i][M.m] = v.tab[i];
	}

	return B;
}

template<typename T>
Matrix<T> operator&(const ColumnVector<T> &v,
		    const Matrix<T> &M)
{
	size_t i, j;

	if (M.n != v.len) {
		throw std::runtime_error(
			"Wrong lengths "
			"for concatenation.");
	}

	Matrix<T> B(M.n, M.m+1);

	for (i = 0; i < M.n; i++) {
		B.tab[i][0] = v.tab[i];
		for (j = 0; j < M.m; j++)
			B.tab[i][j+1] = M.tab[i][j];
	}

	return B;
}

template<typename T>
ColumnVector<T> operator*(const Matrix<T> &M,
			  const ColumnVector<T> &v)
{
	size_t i, j;
	T sum;

	if (M.m != v.len) {
		throw std::runtime_error(
			"Wrong lengths "
			"for multiplication.");
	}

	ColumnVector<T> x(M.n);

	for (i = 0; i < M.n; i++) {
		sum = 0;
		for (j = 0; j < M.m; j++) {
			sum += M[i][j]*v[j];
		}
		x[i] = sum;
	}

	return x;
}

template<typename T>
RowVector<T> operator*(const RowVector<T> &v,
		       const Matrix<T> &M)
{
	size_t i, j;
	T sum;

	if (v.len != M.n) {
		throw std::runtime_error(
			"Wrong lengths "
			"for multiplication.");
	}

	RowVector<T> x(M.m);

	for (j = 0; j < M.m; j++) {
		sum = 0;
		for (i = 0; i < M.n; i++) {
			sum += v[i]*M[i][j];
		}
		x[j] = sum;
	}

	return x;
}

/* Do wygenerowania. */
template void std::swap(RowReference<long double> &&r1,
			RowReference<long double> &&r2);

template class Matrix<long double>;
template class RowReference<long double>;
template std::ostream &operator<<(std::ostream &os,
				  const Matrix<long double> &M);
template Matrix<long double> operator*(Matrix<long double> M1,
				       const Matrix<long double> &M2);
template Matrix<long double> operator+(Matrix<long double> M1,
				       const Matrix<long double> &M2);
template Matrix<long double> operator-(Matrix<long double> M1,
				       const Matrix<long double> &M2);

template
Matrix<long double> operator&(const Matrix<long double> &M,
			      const ColumnVector<long double> &v);
template
Matrix<long double> operator&(const ColumnVector<long double> &v,
			      const Matrix<long double> &M);

template
ColumnVector<long double> operator*(const Matrix<long double> &M,
				    const ColumnVector<long double> &v);
template
RowVector<long double> operator*(const RowVector<long double> &v,
				 const Matrix<long double> &M);
