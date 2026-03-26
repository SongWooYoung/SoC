#ifndef MATRIX_H
#define MATRIX_H


#include <initializer_list>
#include <iostream>
#include <stdexcept> // for std::out_of_range
#include <utility>

class matrix {
private: 
    int rows;
    int cols;
    int *data;

    class RowProxy {
    private:
        int* row_data;
        int cols;

    public:
        RowProxy(int* row_data, int cols) : row_data(row_data), cols(cols) {}

        int& operator[](int col) {
            if (col < 0 || col >= cols) {
                throw std::out_of_range("column index out of range");
            }

            return row_data[col];
        }
    };

    class ConstRowProxy {
    private:
        const int* row_data;
        int cols;

    public:
        ConstRowProxy(const int* row_data, int cols) : row_data(row_data), cols(cols) {}

        const int& operator[](int col) const {
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
    matrix(std::initializer_list<std::initializer_list<int>> values);
    matrix(const matrix& other);
    matrix(matrix&& other) noexcept;
    matrix& operator=(const matrix& other);
    matrix& operator=(matrix&& other) noexcept;
    matrix& operator=(std::initializer_list<std::initializer_list<int>> values);
    ~matrix();

    // operator () overloading
    int& operator()(int row, int col);

    // const version of operator()
    const int& operator()(int row, int col) const;

    RowProxy operator[](int row);
    ConstRowProxy operator[](int row) const;

    int row_count() const {
        return rows;
    }

    int col_count() const {
        return cols;
    }

    int* raw_data() {
        return data;
    }

    const int* raw_data() const {
        return data;
    }


    void transpose();
    void display();

    void shape() const {
        std::cout << "[" << rows << ", " << cols << "]" << std::endl;
    }


};




#endif // MATRIX_H