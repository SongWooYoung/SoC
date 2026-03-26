#ifndef MATRIX_H
#define MATRIX_H


#include <initializer_list>
#include <iostream>
#include <stdexcept> // for std::out_of_range
#include <utility>

namespace {
constexpr int transpose_block_size = 32;
}

template<typename T>
class matrix {
private: 
    int rows;
    int cols;
    T *data;

    class RowProxy {
    private:
        T* row_data;
        int cols;

    public:
        RowProxy(T* row_data, int cols) : row_data(row_data), cols(cols) {}

        T& operator[](int col) {
            if (col < 0 || col >= cols) {
                throw std::out_of_range("column index out of range");
            }

            return row_data[col];
        }
    };

    class ConstRowProxy {
    private:
        const T* row_data;
        int cols;

    public:
        ConstRowProxy(const T* row_data, int cols) : row_data(row_data), cols(cols) {}

        const T& operator[](int col) const {
            if (col < 0 || col >= cols) {
                throw std::out_of_range("column index out of range");
            }

            return row_data[col];
        }
    };

    void allocate();
    void release() noexcept;
    void copy_from(const matrix& other);

public:

    // constructor and destructor
    matrix(int r, int c);
    matrix(std::initializer_list<std::initializer_list<T>> values);
    matrix(const matrix& other);
    matrix(matrix&& other) noexcept;
    matrix& operator=(const matrix& other);
    matrix& operator=(matrix&& other) noexcept;
    matrix& operator=(std::initializer_list<std::initializer_list<T>> values);
    ~matrix();

    // operator () overloading
    T& operator()(int row, int col);

    // const version of operator()
    const T& operator()(int row, int col) const;

    RowProxy operator[](int row);
    ConstRowProxy operator[](int row) const;

    int row_count() const {
        return rows;
    }

    int col_count() const {
        return cols;
    }

    T* raw_data() {
        return data;
    }

    const T* raw_data() const {
        return data;
    }


    void transpose();
    void display();

    void shape() const {
        std::cout << "[" << rows << ", " << cols << "]" << std::endl;
    }


};

// Template implementation

template<typename T>
void matrix<T>::allocate() {
    if (rows == 0 || cols == 0) {
        data = nullptr;
        return;
    }

    data = new T[rows * cols]{};
}

template<typename T>
void matrix<T>::release() noexcept {
    delete[] data;
    data = nullptr;
}

template<typename T>
void matrix<T>::copy_from(const matrix& other) {
    rows = other.rows;
    cols = other.cols;
    allocate();

    for (int index = 0; index < rows * cols; ++index) {
        data[index] = other.data[index];
    }
}

template<typename T>
matrix<T>::matrix(int r, int c) : rows(r), cols(c), data(nullptr) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }

    allocate();
}

template<typename T>
matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> values)
    : rows(static_cast<int>(values.size())), cols(0), data(nullptr) {
    if (rows == 0) {
        return;
    }

    cols = static_cast<int>(values.begin()->size());
    for (const auto& row : values) {
        if (static_cast<int>(row.size()) != cols) {
            throw std::invalid_argument("column count does not match");
        }
    }

    allocate();

    int row_index = 0;
    for (const auto& row : values) {
        int col_index = 0;
        for (const T& value : row) {
            data[row_index * cols + col_index] = value;
            ++col_index;
        }
        ++row_index;
    }
}

template<typename T>
matrix<T>::matrix(const matrix& other) : rows(0), cols(0), data(nullptr) {
    copy_from(other);
}

template<typename T>
matrix<T>::matrix(matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;
}

template<typename T>
matrix<T>& matrix<T>::operator=(const matrix& other) {
    if (this == &other) {
        return *this;
    }

    matrix temp(other);
    std::swap(rows, temp.rows);
    std::swap(cols, temp.cols);
    std::swap(data, temp.data);

    return *this;
}

template<typename T>
matrix<T>& matrix<T>::operator=(matrix&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    rows = other.rows;
    cols = other.cols;
    data = other.data;

    other.rows = 0;
    other.cols = 0;
    other.data = nullptr;

    return *this;
}

template<typename T>
matrix<T>& matrix<T>::operator=(std::initializer_list<std::initializer_list<T>> values) {
    matrix temp(values);
    std::swap(rows, temp.rows);
    std::swap(cols, temp.cols);
    std::swap(data, temp.data);

    return *this;
}

template<typename T>
matrix<T>::~matrix() {
    release();
}

template<typename T>
T& matrix<T>::operator()(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("matrix index out of range");
    }

    return data[row * cols + col];
}

template<typename T>
const T& matrix<T>::operator()(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("matrix index out of range");
    }

    return data[row * cols + col];
}

template<typename T>
typename matrix<T>::RowProxy matrix<T>::operator[](int row) {
    if (row < 0 || row >= rows) {
        throw std::out_of_range("row index out of range");
    }

    return RowProxy(data + row * cols, cols);
}

template<typename T>
typename matrix<T>::ConstRowProxy matrix<T>::operator[](int row) const {
    if (row < 0 || row >= rows) {
        throw std::out_of_range("row index out of range");
    }

    return ConstRowProxy(data + row * cols, cols);
}

template<typename T>
void matrix<T>::transpose() {
    if (rows == cols) {
        for (int row_block = 0; row_block < rows; row_block += transpose_block_size) {
            int row_limit = row_block + transpose_block_size;
            if (row_limit > rows) {
                row_limit = rows;
            }

            for (int col_block = row_block; col_block < cols; col_block += transpose_block_size) {
                int col_limit = col_block + transpose_block_size;
                if (col_limit > cols) {
                    col_limit = cols;
                }

                if (row_block == col_block) {
                    for (int row = row_block; row < row_limit; ++row) {
                        int row_offset = row * cols;
                        int diagonal_start = row + 1;
                        if (diagonal_start < col_block) {
                            diagonal_start = col_block;
                        }

                        for (int col = diagonal_start; col < col_limit; ++col) {
                            std::swap(data[row_offset + col], data[col * cols + row]);
                        }
                    }
                    continue;
                }

                for (int row = row_block; row < row_limit; ++row) {
                    int row_offset = row * cols;
                    for (int col = col_block; col < col_limit; ++col) {
                        std::swap(data[row_offset + col], data[col * cols + row]);
                    }
                }
            }
        }
        return;
    }

    matrix temp(cols, rows);
    for (int row_block = 0; row_block < rows; row_block += transpose_block_size) {
        int row_limit = row_block + transpose_block_size;
        if (row_limit > rows) {
            row_limit = rows;
        }

        for (int col_block = 0; col_block < cols; col_block += transpose_block_size) {
            int col_limit = col_block + transpose_block_size;
            if (col_limit > cols) {
                col_limit = cols;
            }

            for (int row = row_block; row < row_limit; ++row) {
                int source_offset = row * cols;
                for (int col = col_block; col < col_limit; ++col) {
                    temp.data[col * rows + row] = data[source_offset + col];
                }
            }
        }
    }

    *this = std::move(temp);
}

template<typename T>
void matrix<T>::display() {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            std::cout << data[row * cols + col] << ' ';
        }
        std::cout << '\n';
    }
}




#endif // MATRIX_H