#ifndef IMAGE_H
#define IMAGE_H 

#include <fdeep/fdeep.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <queue>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <type_traits>

namespace fs = std::filesystem;

enum class Color { black = 0, white = 255, gray = 120};
enum class Direction { left = 0, right = 1, up = 2, down = 3};

struct pixel {
    pixel(int j, int i)
        : j(j)
        , i(i)
    {
    }

    bool operator==(pixel other) const
    {
        return j == other.j && i == other.i;
    }

    bool operator!=(pixel other) const
    {
        return !(*this == other);
    }

    pixel next_pixel(Direction dir) const
    {
        return (dir == Direction::right) ? pixel{j, i+1} :
               (dir == Direction::left) ? pixel{j, i-1} :
               (dir == Direction::up) ? pixel{j-1, i} :
                pixel{j+1, i};
    }

    pixel previous_pixel(Direction dir) const
    {
        return (dir == Direction::right) ? pixel{j, i-1} :
               (dir == Direction::left) ? pixel{j, i+1} :
               (dir == Direction::up) ? pixel{j+1, i} :
                pixel{j-1, i};
    }

    struct hash
    {
      size_t operator()(pixel p) const
      {
        return std::hash<int>()(p.j) ^ (std::hash<int>()(p.i) << 1);
      }
    };

    int j, i;
};

struct borders {
    // [left, right) x [top, bottom)
    borders(int left, int right, int top, int bottom)
        : m_left(left)
        , m_right(right)
        , m_top(top)
        , m_bottom(bottom)
    {
    }

    void update(pixel p)
    {
        // try to expande borders with new pixel
        m_left = std::min(p.i, m_left);
        m_right = std::max(p.i+1, m_right);
        m_top = std::min(p.j, m_top);
        m_bottom = std::max(p.j+1, m_bottom);
    }

    int width() const
    {
        return m_right - m_left;
    }

    int height() const
    {
        return m_bottom - m_top;
    }

    int left() const
    {
        return m_left;
    }

    int right() const
    {
        return m_right;
    }

    int top() const
    {
        return m_top;
    }

    int bottom() const
    {
        return m_bottom;
    }

private:
    int m_left, m_right, m_top, m_bottom;
};

class image {
public:
    image() = default;
    image(const fs::path& path);
    // create a white image
    image(int height, int width);
    // create an image from image columns [left, right)
    image(const image& img, int left, int right);
    // this will not copy image data
    image(const image& img) = default;
    // this will not copy image data
    image& operator=(const image& img) = default;
    image(image&& other) = default;
    image& operator=(image&& img) = default;
    image(cv::Mat data);
    ~image() = default;

    fdeep::tensor get_tensor(float min, float max) const;

    void rotate(double angle);

    static image copy(const image& img)
    {
        return img.mat().clone();
    }

    static image copy(const image& img, int left, int right)
    {
        return img.mat()(cv::Range::all(), cv::Range(left, right)).clone();
    }

    int rows() const;
    int cols() const;

    void crop();
    // add padding of 1
    void add_border(int padding = 1);

    void add_border(Direction d, int padding);

    std::pair<int,int> dim() const;

    const cv::Mat& mat() const;
    cv::Mat& mat();

    void blur(int filter_size);

    void show() const;
    void save(const fs::path& path) const;

    void resize(int width, int height);

    unsigned char& operator()(int j, int i);
    const unsigned char& operator()(int j, int i) const;

    unsigned char& operator()(pixel p);
    const unsigned char& operator()(pixel p) const;

    unsigned char& color_at(int j, int i);
    const unsigned char& color_at(int j, int i) const;

    unsigned char& color_at(pixel p);
    const unsigned char& color_at(pixel p) const;

    bool check_color(pixel p, Color color) const;

    bool in_range(pixel p) const;

    borders bfs(pixel p, std::unordered_set<pixel, pixel::hash>& visited) const;

    // width, height
    std::pair<int,int> get_component_avg_size_and_remove_noise();

    // used for images of height 1
    int sum(int start, int end) const;

    void fill_row(int row, int value);
    void fill_row(int row, Color color);
    void fill(int top, int bottom, Color color);

    bool row_empty(int row, int starti, int endi) const;
    bool column_empty(int column, int startj, int endj) const;

    void threshold(bool otsu = false);
    void threshold(int x);

    image concatenate_horizontal(const image& other) const;
    image operator+(const image& other) const;
private:
    cv::Mat m_data;
};


// This part of the code is copied from https://gitlab.com/Queuecumber/opencvit and modified.
namespace cvit
{
template<class AttrType, class MatType, class CRTP, bool is_const_iterator, bool reversed>
class attr_iterator : public std::iterator<std::forward_iterator_tag, MatType>
{
public:
    typedef typename std::conditional<is_const_iterator, const MatType, MatType>::type MatMutType;

    attr_iterator(void) = default;

    attr_iterator(const attr_iterator& it) = default;

    attr_iterator(image mat, AttrType loc = 0)
        : mat(std::move(mat)), pos(loc), endGuard(false)
    { }

    CRTP &operator++(void)
    {
        if (!reversed) {
            increment_attribute(pos);
        }
        else {
            decrement_attribute(pos);
        }
        return static_cast<CRTP &>(*this);
    }

    CRTP operator++(int)
    {
        CRTP tmp = static_cast<CRTP &>(*this);
        if (!reversed) {
            increment_attribute(pos);
        }
        else {
            decrement_attribute(pos);
        }
        return tmp;
    }

    bool operator==(const attr_iterator& other)
    {
        if (other.endGuard)
        {
            return (!reversed) ? pos >= get_termination_condition(mat) : pos < 0;
        }
        else if (endGuard)
        {
            return (!reversed) ? other.pos >= get_termination_condition(other.mat) : other.pos < 0;
        }
        else
        {
            if (mat.mat().data != other.mat.mat().data) {
                throw std::runtime_error("Iterator matrices are different");
            }

            return other.pos == pos;
        }
    }

    bool operator!=(const attr_iterator& other)
    {
        return !(*this == other);
    }

    MatMutType operator*(void)
    {
        return get_submat(mat, pos);
    }

protected:
    virtual void increment_attribute(AttrType &) = 0;
    virtual void decrement_attribute(AttrType &) = 0;
    virtual MatType get_submat(const image&, AttrType) = 0;
    virtual AttrType get_termination_condition(const image&) = 0;

private:
    image mat;
    AttrType pos;
    bool endGuard = true;
};

template<class MatType, class CRTP, bool is_const_iterator, bool reversed>
class rc_iterator_base : public attr_iterator<int, MatType, CRTP, is_const_iterator, reversed>
{
public:
    using attr_iterator<int, MatType, CRTP, is_const_iterator, reversed>::attr_iterator;

    rc_iterator_base(void) = default;

    rc_iterator_base(image mat, int loc, int step)
        : attr_iterator<int, MatType, CRTP, is_const_iterator, reversed>(std::move(mat), loc), step(step)
    { 
    }

protected:
    virtual void increment_attribute(int &p) 
    {
        p += step;
    }

    virtual void decrement_attribute(int &p) 
    {
        p -= step;
    }

    virtual MatType get_submat(const image& m, int c)  = 0;
    virtual int get_termination_condition(const image& m) = 0;

private:
    int step = 1;
};

template<class MatType, class CRTP, bool is_const_iterator, bool reversed>
class column_iterator_base : public rc_iterator_base<MatType, CRTP, is_const_iterator, reversed>
{
public:
    using rc_iterator_base<MatType, CRTP, is_const_iterator, reversed>::rc_iterator_base;

protected:
    virtual MatType get_submat(const image& m, int c)  = 0;

    virtual int get_termination_condition(const image& m) { return m.cols(); }
};

template<class MatType, class CRTP, bool is_const_iterator, bool reversed>
class row_iterator_base : public rc_iterator_base<MatType, CRTP, is_const_iterator, reversed>
{
public:
    using rc_iterator_base<MatType, CRTP, is_const_iterator, reversed>::rc_iterator_base;

protected:
    virtual MatType get_submat(const image& m, int c) = 0;

    virtual int get_termination_condition(const image& m) { return m.rows(); }
};

template<bool is_const_iterator, bool reversed>
class column_mat_iterator_base : public column_iterator_base<image, column_mat_iterator_base<is_const_iterator, reversed>, is_const_iterator, reversed>
{
public:
    using column_iterator_base<image, column_mat_iterator_base<is_const_iterator, reversed>, is_const_iterator, reversed>::column_iterator_base;

protected:
    virtual image get_submat(const image& m, int c) { return m.mat().col(c); }
};

template<bool is_const_iterator, bool reversed>
class row_mat_iterator_base : public row_iterator_base<image, row_mat_iterator_base<is_const_iterator, reversed>, is_const_iterator, reversed>
{
public:
    using row_iterator_base<image, row_mat_iterator_base<is_const_iterator, reversed>, is_const_iterator, reversed>::row_iterator_base;

protected:
    virtual image get_submat(const image& m, int c) { return m.mat().row(c); }
};

typedef column_mat_iterator_base<false, false> column_iterator;
typedef column_mat_iterator_base<true, false> const_column_iterator;
typedef column_mat_iterator_base<false, true> column_iterator_r;

typedef row_mat_iterator_base<false, false> row_iterator;
typedef row_mat_iterator_base<true, false> const_row_iterator;
typedef row_mat_iterator_base<false, true> row_iterator_r;
}

#endif /* IMAGE_H */
