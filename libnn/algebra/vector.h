#ifndef VECTOR_H_
#define VECTOR_H_

#include <ostream>
#include <cstddef>
#include <memory>
#include <initializer_list>
#include <iterator>

template<typename T>
class BasicVector;

template<typename T>
class RowReference;

template<typename T>
class RowVector;

template<typename T>
class ColumnVector;

template<typename T>
class Vector;

#include "matrix.h"

template<typename T>
class BasicVector {
	static_assert(std::is_arithmetic<T>::value, "");

protected:
	size_t len;
	T *tab;

	BasicVector(T *tab, size_t len);

public:
	BasicVector(void) = delete;
	virtual ~BasicVector(void) {};

	size_t length(void) const;

	T &operator[](const int i);
	const T &operator[](const int i) const;

	class iterator;
	class const_iterator;
	
	constexpr iterator begin(void) noexcept { return iterator(&tab[0]); }
	constexpr const_iterator begin(void) const noexcept { return const_iterator(&tab[0]); }
	constexpr const_iterator cbegin(void) const noexcept { return const_iterator(&tab[0]); }

	constexpr iterator end(void) noexcept { return iterator(&tab[len]); }
	constexpr const_iterator end(void) const noexcept { return const_iterator(&tab[len]); }
	constexpr const_iterator cend(void) const noexcept { return const_iterator(&tab[len]); }

	friend class RowVector<T>;
	friend class RowReference<T>;
};

template<typename T>
class BasicVector<T>::iterator {
	using iterator_category = std::bidirectional_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type	= T;
	using pointer		= T*;
	using reference		= T&;

private:
	pointer m_ptr;

public:
	iterator(pointer ptr) noexcept : m_ptr(ptr) {}

	reference operator*() const { return *m_ptr; }
	pointer operator->() { return m_ptr; }

	iterator &operator++() { m_ptr++; return *this; }
	iterator operator++(int) { iterator tmp = *this;
				   ++(*this);
				   return tmp; }
	iterator &operator--() { m_ptr--; return *this; }
	iterator operator--(int) { iterator tmp = *this;
				   --(*this);
				   return tmp; }

	friend bool operator==(const iterator &a, const iterator &b)
					{ return a.m_ptr == b.m_ptr; };
	friend bool operator!=(const iterator &a, const iterator &b)
					{ return a.m_ptr != b.m_ptr; };
};

template<typename T>
class BasicVector<T>::const_iterator {
	using iterator_category = std::bidirectional_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type	= const T;
	using pointer		= const T*;
	using reference		= const T&;

private:
	pointer m_ptr;

public:
	const_iterator(pointer ptr) noexcept : m_ptr(ptr) {}

	reference operator*() const { return *m_ptr; }
	pointer operator->() { return m_ptr; }

	const_iterator &operator++() { m_ptr++; return *this; }
	const_iterator operator++(int) { const_iterator tmp = *this;
				   ++(*this);
				   return tmp; }
	const_iterator &operator--() { m_ptr--; return *this; }
	const_iterator operator--(int) { const_iterator tmp = *this;
				   --(*this);
				   return tmp; }

	friend bool operator==(const const_iterator &a, const const_iterator &b)
					{ return a.m_ptr == b.m_ptr; };
	friend bool operator!=(const const_iterator &a, const const_iterator &b)
					{ return a.m_ptr != b.m_ptr; };
};

template<typename T>
class Vector : public BasicVector<T> {
public:
	Vector(size_t len);
	Vector(std::initializer_list<T> c);
	Vector(const Vector&);
	Vector(Vector&&);
	Vector(const RowVector<T> &v) :
	      Vector(static_cast<const Vector&>(v)) {};
	Vector(RowVector<T> &&v) :
	      Vector(static_cast<Vector&&>(v)) {};
	Vector(const ColumnVector<T> &v) :
	      Vector(static_cast<const Vector&>(v)) {};
	Vector(ColumnVector<T> &&v) :
	      Vector(static_cast<Vector&&>(v)) {};
	Vector(RowReference<T>&&);
	~Vector(void) override;
};

template<typename T>
class RowVector : public Vector<T> {
public:
	RowVector(size_t len) : Vector<T>(len) {};
	RowVector(std::initializer_list<T> c) : Vector<T>(c) {};
	RowVector(const Vector<T> &v) : Vector<T>(v) {};
	RowVector(Vector<T> &&v) : Vector<T>(v) {};
	RowVector(const RowVector<T> &v) : Vector<T>(v) {};
	RowVector(RowVector<T> &&v) : Vector<T>(v) {};
	RowVector(RowReference<T> &&v) : Vector<T>(std::move(v)) {};
	~RowVector(void) override = default;

	ColumnVector<T> transpose(void) const;

	template<typename U> friend
	std::ostream &operator<<(std::ostream&,
				 const RowVector<U>&);

	RowVector &operator=(const RowVector &v) &;
	RowVector &operator=(RowVector &&v) &;

	RowVector &operator+=(const RowVector &v) &;
	RowVector &operator-=(const RowVector &v) &;
	RowVector &operator+=(const RowReference<T> &&v) &;
	RowVector &operator-=(const RowReference<T> &&v) &;

	template<typename U> friend
	U operator*(const RowVector<U> &v1, const ColumnVector<U> &v2);
	template<typename U> friend
	RowVector<U> operator*(U c, RowVector<U> v);
	template<typename U> friend
	RowVector<U> operator+(RowVector<U> v1,
			       const RowVector<U> &v2);
	template<typename U> friend
	RowVector<U> operator-(RowVector<U> v1,
			       const RowVector<U> &v2);

	template<typename U> friend
	RowReference<U> operator+(RowReference<U> v1,
				  const RowVector<U> &v2);
	template<typename U> friend
	RowVector<U> operator+(RowVector<U> v1,
			       const RowReference<U> &&v2);

	template<typename U> friend
	ColumnVector<U> operator*(const Matrix<U> &M,
				  const ColumnVector<U> &v);
	template<typename U> friend
	RowVector<U> operator*(const RowVector<U> &v,
			       const Matrix<U> &M);
};

template<typename T>
class ColumnVector : public Vector<T> {
public:
	ColumnVector(size_t len) : Vector<T>(len) {};
	ColumnVector(std::initializer_list<T> c) : Vector<T>(c) {};
	ColumnVector(const Vector<T> &v) : Vector<T>(v) {};
	ColumnVector(Vector<T> &&v) : Vector<T>(v) {};
	ColumnVector(const ColumnVector<T> &v) : Vector<T>(v) {};
	ColumnVector(ColumnVector<T> &&v) : Vector<T>(v) {};
	ColumnVector(RowReference<T> &&v) : Vector<T>(std::move(v)) {};
	~ColumnVector(void) override = default;

	RowVector<T> transpose(void) const;

	template<typename U> friend
	std::ostream &operator<<(std::ostream&,
				 const ColumnVector<U>&);

	ColumnVector &operator=(const ColumnVector &v) &;
	ColumnVector &operator=(ColumnVector &&v) &;

	ColumnVector &operator+=(const ColumnVector &v) &;
	ColumnVector &operator-=(const ColumnVector &v) &;

	template<typename U> friend
	U operator*(const RowVector<U> &v1, const ColumnVector<U> &v2);
	template<typename U> friend
	ColumnVector<U> operator*(U c, ColumnVector<U> v);
	template<typename U> friend
	ColumnVector<U> operator+(ColumnVector<U> v1,
				  const ColumnVector<U> &v2);
	template<typename U> friend
	ColumnVector<U> operator-(ColumnVector<U> v1,
				  const ColumnVector<U> &v2);

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

#endif /* VECTOR_H_ */
