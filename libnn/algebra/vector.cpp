#include <cmath>

#include "vector.h"
#include "matrix.h"

/* Definicje szablonów. */

template<typename T>
BasicVector<T>::BasicVector(T *tab, size_t len) :
len(len),
tab(tab)
{
}

template<typename T>
size_t BasicVector<T>::length(void) const
{
	return len;
}

template<typename T>
T &BasicVector<T>::operator[](const int i)
{
	return tab[i];
}

template<typename T>
const T &BasicVector<T>::operator[](const int i) const
{
	return tab[i];
}

template<typename T>
Vector<T>::Vector(std::initializer_list<T> c) :
BasicVector<T>(new T[c.size()], c.size())
{
	if (c.size() < 1) {
		throw std::runtime_error(
			"Vector length have "
			"to be positive integer.");
	}

	std::copy(c.begin(), c.end(), this->tab);
}

template<typename T>
Vector<T>::Vector(size_t len) :
BasicVector<T>(new T[len], len)
{
	if (len < 1) {
		throw std::runtime_error(
			"Vector length have "
			"to be positive integer.");
	}
}

template<typename T>
Vector<T>::Vector(const Vector &v) :
BasicVector<T>(new T[v.len], v.len)
{
	size_t i;

	for (i = 0; i < this->len; i++)
		this->tab[i] = v.tab[i];
}

template<typename T>
Vector<T>::Vector(Vector &&v) :
BasicVector<T>(v.tab, v.len)
{
	v.tab = nullptr;
}

template<typename T>
Vector<T>::Vector(RowReference<T> &&v) :
BasicVector<T>(new T[v.len], v.len)
{
	size_t i;

	for (i = 0; i < this->len; i++)
		this->tab[i] = v.tab[i];
}

template<typename T>
T norm(const BasicVector<T> &v)
{
	T sum ;

	sum = 0;
	for (typename BasicVector<T>::const_iterator it = v.cbegin();
	     it != v.cend(); it++) {
		sum += (*it)*(*it);
	}

	return std::sqrt(sum);
}

template<typename T>
Vector<T>::~Vector(void)
{
	delete[] this->tab;
}

template<typename T>
ColumnVector<T> RowVector<T>::transpose(void) const
{
	return ColumnVector(*this);
}

template<typename T>
RowVector<T> &RowVector<T>::operator+=(const RowVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] += v.tab[i];

	return *this;
}

template<typename T>
RowVector<T> &RowVector<T>::operator-=(const RowVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] -= v.tab[i];

	return *this;
}

template<typename T>
RowVector<T> &RowVector<T>::operator+=(const RowReference<T> &&v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] += v.tab[i];

	return *this;
}

template<typename T>
RowVector<T> &RowVector<T>::operator-=(const RowReference<T> &&v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] -= v.tab[i];

	return *this;
}

template<typename T>
std::ostream &operator<<(std::ostream &os,
			 const RowVector<T> &V)
{
	size_t i;
	size_t len = V.len;

	if (len == 0) {
		os << "[]";
		return os;
	}

	if (len == 1) {
		os << '[' << V[0] << ']';
		return os;
	}

	os << '[' << V[0] << ", ";
	for (i = 1; i < len-1; i++)
		os << V[i] << ", ";
	os << V[len-1] << ']';

	return os;
}

template<typename T>
RowVector<T> &RowVector<T>::operator=(const RowVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Vectors are of different "
			"lengths.");
	}

	for (i = 0; i < this->len; i++)
		this->tab[i] = v.tab[i];

	return *this;
}

template<typename T>
RowVector<T> &RowVector<T>::operator=(RowVector<T> &&v) &
{
	if (this->len != v.len) {
		throw std::runtime_error(
			"Vectors are of different "
			"lengths.");
	}

	this->tab = v.tab;
	v.tab = nullptr;

	return *this;
}

template<typename T>
RowVector<T> ColumnVector<T>::transpose(void) const
{
	return RowVector(*this);
}

template<typename T>
ColumnVector<T> &ColumnVector<T>::operator+=(const ColumnVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] += v.tab[i];

	return *this;
}

template<typename T>
ColumnVector<T> &ColumnVector<T>::operator-=(const ColumnVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Wrong Vector lengths "
			"for addition.");
	}

	for (i = 0; i < v.len; i++)
		this->tab[i] -= v.tab[i];

	return *this;
}

template<typename T>
std::ostream &operator<<(std::ostream &os,
			 const ColumnVector<T> &V)
{
	size_t i;
	size_t len = V.len;

	if (len == 0) {
		os << "[]";
		return os;
	}

	if (len == 1) {
		os << '[' << V[0] << ']';
		return os;
	}

	os << '[' << V[0] << ",\n";
	for (i = 1; i < len-1; i++)
		os << ' ' << V[i] << ",\n";
	os << ' ' << V[len-1] << ']';

	return os;
}

template<typename T>
ColumnVector<T> &ColumnVector<T>::operator=(const ColumnVector<T> &v) &
{
	size_t i;

	if (this->len != v.len) {
		throw std::runtime_error(
			"Vectors are of different "
			"lengths.");
	}

	for (i = 0; i < this->len; i++)
		this->tab[i] = v.tab[i];

	return *this;
}

template<typename T>
ColumnVector<T> &ColumnVector<T>::operator=(ColumnVector<T> &&v) &
{
	if (this->len != v.len) {
		throw std::runtime_error(
			"Vectors are of different "
			"lengths.");
	}

	this->tab = v.tab;
	v.tab = nullptr;

	return *this;
}

template <typename T>
T operator*(const RowVector<T> &v1, const ColumnVector<T> &v2)
{
	size_t i;
	long double result;

	result = 0;
	for (i = 0; i < v1.length(); i++)
		result += v1[i] * v2[i];

	return result;
}

template <typename T>
RowVector<T> operator*(T c, RowVector<T> v)
{
	for (typename RowVector<T>::iterator it = v.begin();
	     it != v.end(); it++) {
		(*it) *= c;
	}

	return v;
}

template <typename T>
ColumnVector<T> operator*(T c, ColumnVector<T> v)
{
	for (typename ColumnVector<T>::iterator it = v.begin();
	     it != v.end(); it++) {
		(*it) *= c;
	}

	return v;
}

template <typename T>
RowVector<T> operator+(RowVector<T> v1, const RowVector<T> &v2)
{
	return (v1 += v2);
}

template <typename T>
RowReference<T> operator+(RowReference<T> v1, const RowVector<T> &v2)
{
	return (RowVector(v1) += v2);
}

template <typename T>
RowVector<T> operator+(RowVector<T> v1, const RowReference<T> &&v2)
{
	return (v1 += v2);
}

template <typename T>
ColumnVector<T> operator-(ColumnVector<T> v1,
		          const ColumnVector<T> &v2)
{
	return (v1 -= v2);
}

/* Do wygenerowania. */
template class BasicVector<long double>;
template class Vector<long double>;
template class RowVector<long double>;
template class ColumnVector<long double>;

template long double norm(const BasicVector<long double> &v);

template std::ostream &operator<<(std::ostream &os,
				  const RowVector<long double> &V);
template std::ostream &operator<<(std::ostream &os,
				  const ColumnVector<long double> &V);

template long double operator*(const RowVector<long double> &v1, const ColumnVector<long double> &v2);
template RowVector<long double> operator*(long double c, RowVector<long double> v);
template ColumnVector<long double> operator*(long double c, ColumnVector<long double> v);
template RowVector<long double> operator+(RowVector<long double> v1,
					  const RowVector<long double> &v2);
template ColumnVector<long double>
operator-(ColumnVector<long double> v1,
	  const ColumnVector<long double> &v2);
