#include "image.h" 
#include <cassert>

// image
image::image(const fs::path& path)
    : m_data((fs::exists(path)) ? cv::imread(path.string(), cv::IMREAD_GRAYSCALE) : (throw std::runtime_error("File not found."), cv::Mat()))
{
}

image::image(int height, int width)
    : m_data(cv::Mat(height, width, CV_8UC1, cv::Scalar(static_cast<int>(Color::white))))
{
}

image::image(const image& img, int left, int right)
    : m_data(img.mat()(cv::Range::all(), cv::Range(left, right)))
{
}

image::image(cv::Mat data)
    : m_data(std::move(data))
{
}

fdeep::tensor image::get_tensor(float low, float high) const
{
    assert(m_data.isContinuous());
    // Use the correct scaling, i.e., low and high
    return fdeep::tensor_from_bytes(m_data.ptr(),
        static_cast<std::size_t>(m_data.rows),
        static_cast<std::size_t>(m_data.cols),
        static_cast<std::size_t>(m_data.channels()), 
        low,
        high);
}

int image::rows() const
{
    return m_data.rows;
}

int image::cols() const
{
    return m_data.cols;
}

void image::resize(int width, int height)
{
    cv::resize(m_data, m_data, cv::Size(width, height));
}

void image::crop()
{
    auto [h, w] = dim();
    auto not_white = [](const auto& x) { return x(0,0) != static_cast<unsigned char>(Color::white); };

    auto top_it = std::find_if(cvit::row_iterator(*this), cvit::row_iterator(), [&](const auto& row) {
        return std::any_of(cvit::column_iterator(row), cvit::column_iterator(), not_white);
    });
    int top = (top_it != cvit::row_iterator()) ? std::distance(cvit::row_iterator(*this), top_it) : -1;

    auto bottom_it = std::find_if(cvit::row_iterator_r(*this, h-1), cvit::row_iterator_r(), [&](const auto& row) {
        return std::any_of(cvit::column_iterator(row), cvit::column_iterator(), not_white);
    });
    int bottom = (bottom_it != cvit::row_iterator_r()) ? std::distance(bottom_it, cvit::row_iterator_r(*this,0)) : -1;

    auto left_it = std::find_if(cvit::column_iterator(*this), cvit::column_iterator(), [&](const auto& col) {
        return std::any_of(cvit::row_iterator(col), cvit::row_iterator(), not_white);
    });
    int left = (left_it != cvit::column_iterator()) ? std::distance(cvit::column_iterator(*this), left_it) : -1;

    auto right_it = std::find_if(cvit::column_iterator_r(*this, w-1), cvit::column_iterator_r(), [&](const auto& col) {
        return std::any_of(cvit::row_iterator(col), cvit::row_iterator(), not_white);
    });
    int right = (right_it != cvit::column_iterator_r()) ? std::distance(right_it, cvit::column_iterator_r(*this, 0)) : -1;

    if (left == -1 || right == -1 || top == -1 || bottom == -1) {
        // cannot crop an empty picture
        return;
    }

    borders b{left, right+1, top, bottom+1};
    m_data = m_data({b.left(), b.top(), b.width(), b.height()});
}

void image::add_border(int padding)
{
    cv::copyMakeBorder(m_data, m_data, padding, padding, padding, padding, cv::BORDER_CONSTANT, static_cast<int>(Color::white));
}

void image::add_border(Direction d, int padding)
{
    if (padding <= 0) {
        return;
    }
    int top = (d == Direction::up) ? padding : 0;
    int bottom = (d == Direction::down) ? padding : 0;
    int left = (d == Direction::left) ? padding : 0;
    int right = (d == Direction::right) ? padding : 0;
    cv::copyMakeBorder(m_data, m_data, top, bottom, left, right, cv::BORDER_CONSTANT, static_cast<int>(Color::white));
}

std::pair<int,int> image::dim() const
{
    return {m_data.rows, m_data.cols};
}

const cv::Mat& image::mat() const
{
    return m_data;
}

cv::Mat& image::mat()
{
    return m_data;
}

void image::show() const
{
    imshow("Display window", m_data);
}

void image::save(const fs::path& path) const
{
    if (!path.empty() && !m_data.empty()) {
        imwrite(path.string(), m_data);
    }
}

unsigned char& image::operator()(int j, int i)
{
    return m_data.at<unsigned char>(j,i);
}

const unsigned char& image::operator()(int j, int i) const
{
    return m_data.at<unsigned char>(j,i);
}

unsigned char& image::operator()(pixel p)
{
    return m_data.at<unsigned char>(p.j,p.i);
}

const unsigned char& image::operator()(pixel p) const
{
    return m_data.at<unsigned char>(p.j,p.i);
}

unsigned char& image::color_at(int j, int i)
{
    return (*this)(j,i);
}

const unsigned char& image::color_at(int j, int i) const
{
    return (*this)(j,i);
}

unsigned char& image::color_at(pixel p)
{
    return (*this)(p);
}

const unsigned char& image::color_at(pixel p) const
{
    return (*this)(p);
}

bool image::check_color(pixel p, Color color) const
{
    return in_range(p) && color_at(p) == static_cast<unsigned char>(color);
}

bool image::in_range(pixel p) const
{
    return (p.j < m_data.rows && p.j >= 0 && p.i < m_data.cols && p.i >= 0);
}

borders image::bfs(pixel p, std::unordered_set<pixel, pixel::hash>& visited) const
{
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                // if neighbour is black and not yet visited
                if (check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    return b;
}

// width, height
std::pair<int,int> image::get_component_avg_size_and_remove_noise()
{
    std::unordered_set<pixel, pixel::hash> visited;
    // width, height
    std::pair<int,int> size_sum{0,0};
    int components_count = 0;

    for (int j = 0; j < rows(); ++j) {
        for (int i = 0; i < cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                int visited_before = visited.size();
                borders b = bfs(p, visited);
                // remove noise
                if (visited.size() - visited_before < 3) {
                    for (int jb = b.top(); jb < b.bottom(); ++jb) {
                        for (int ib = b.left(); ib < b.right(); ++ib) {
                            color_at(jb, ib) = static_cast<unsigned char>(Color::white);
                        }
                    }
                }
                else {
                    std::pair<int,int> this_s{b.width(), b.height()};
                    if (this_s.first >= 5 && this_s.second >= 5) {
                        size_sum.first += this_s.first;
                        size_sum.second += this_s.second;
                        ++components_count;
                    }
                }
            }
        }
    }

    return {size_sum.first/components_count, size_sum.second/components_count};
}

int image::sum(int begin, int end) const
{
    assert(rows() == 1);
    return std::accumulate(cvit::column_iterator(*this, begin), cvit::column_iterator(*this, end), 0, [](int s, const auto& x) { return s + x(0,0); });
}

void image::fill_row(int row, int value)
{
    auto row_it = cvit::row_iterator(*this, row);
    std::for_each(cvit::column_iterator(*row_it), cvit::column_iterator(), [&](auto&& x) { x(0,0) = value; });
}

void image::fill_row(int row, Color color)
{
    auto row_it = cvit::row_iterator(*this, row);
    std::for_each(cvit::column_iterator(*row_it), cvit::column_iterator(), [&](auto&& x) { x(0,0) = static_cast<unsigned char>(color); });
}

void image::fill(int top, int bottom, Color color)
{
    std::for_each(cvit::row_iterator(*this, top), cvit::row_iterator(*this, bottom), [&](const auto& row) {
        std::for_each(cvit::column_iterator(row), cvit::column_iterator(), [&](auto&& x) { x(0,0) = static_cast<unsigned char>(color); });
    });
}

bool image::row_empty(int row, int start, int end) const
{
    auto row_it = cvit::row_iterator(*this, row);
    return std::none_of(cvit::column_iterator(*row_it, start), cvit::column_iterator(*row_it, end), 
            [](const auto& x) { return x(0,0) == static_cast<int>(Color::black); });
}

bool image::column_empty(int column, int start, int end) const
{
    auto column_it = cvit::column_iterator(*this, column);
    return std::none_of(cvit::row_iterator(*column_it, start), cvit::row_iterator(*column_it, end),
            [](const auto& x) { return x(0,0) == static_cast<unsigned char>(Color::black); });
}

void image::threshold(bool otsu)
{
    if (otsu) {
        cv::threshold(m_data, m_data, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }
    else {
        cv::threshold(m_data, m_data, 200, 255, cv::THRESH_BINARY);
    }
}

void image::threshold(int x)
{
    for (int j = 0; j < rows(); ++j) {
        for (int i = 0; i < cols(); ++i) {
            color_at(j,i) = (color_at(j,i) > x) ? static_cast<unsigned char>(Color::white) : static_cast<unsigned char>(Color::black);
        }
    }
}

image image::operator+(const image& other) const
{
    assert(rows() == other.rows());
    cv::Mat result;
    cv::hconcat(m_data, other.m_data, result);
    return image(std::move(result));
}

image image::concatenate_horizontal(const image& other) const
{
    assert(cols() == other.cols());
    cv::Mat result;
    cv::vconcat(other.m_data,m_data, result);
    return image(std::move(result));
}

void image::blur(int filter_size)
{
    cv::GaussianBlur(m_data, m_data, cv::Size(filter_size, filter_size), 0, 0);
}

void image::rotate(double angle)
{
    // get rotation matrix for rotating the image around its center in pixel coordinates
    cv::Point2f center((m_data.cols-1)/2.0, (m_data.rows-1)/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, center not relevant
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), m_data.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - m_data.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - m_data.rows/2.0;

    cv::warpAffine(m_data, m_data, rot, bbox.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}
