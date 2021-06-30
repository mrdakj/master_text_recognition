#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <fdeep/fdeep.hpp>
#include "include/image.h"
#include "include/line_segmentation.h"

const auto model_one = fdeep::load_model("../model/one.json", true, fdeep::dev_null_logger);
const auto model_one_two = fdeep::load_model("../model/one_two.json", true, fdeep::dev_null_logger);
const auto model_first = fdeep::load_model("../model/first.json", true, fdeep::dev_null_logger);
const auto model_second = fdeep::load_model("../model/second.json", true, fdeep::dev_null_logger);

std::unordered_set<std::string> dictionary;

void load_dictionary()
{
    std::ifstream infile("../dictionary/google-10000-english.txt");
    std::string word;
    while (std::getline(infile, word))
    {
        dictionary.insert(word);
    }
    std::cout << "loading dictionary done" << std::endl;
}

std::pair<std::string, std::string> recognize(image& component);

std::pair<std::vector<pixel>, borders> bfs(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited)
{
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    std::vector<pixel> black;

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        black.push_back(current_pixel);
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                // if neighbour is black and not yet visited
                if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    return {black, b};
}

bool is_dot(const image& img)
{
    return (img.rows() < 20 && img.cols() < 20);
}

void glue(std::pair<borders,image>& component, std::pair<borders,image>& dot)
{
    component.second.add_border(Direction::right, dot.first.right() - component.first.right());
    component.second.add_border(Direction::left, component.first.left() - dot.first.left());
    dot.second.add_border(Direction::right,  component.first.right() - dot.first.right());
    dot.second.add_border(Direction::left,  dot.first.left() - component.first.left());

    component.second.add_border(Direction::up, component.first.top() - dot.first.bottom());

    component.second = component.second.concatenate_horizontal(dot.second);

    component.first = borders(
            std::min(component.first.left(), dot.first.left()),
            std::max(component.first.right(), dot.first.right()),
            dot.first.top(),
            component.first.bottom());
}

void get_components(const image& img, int line_num)
{
    fs::create_directory("../out/" + std::to_string(line_num));

    std::unordered_set<pixel, pixel::hash> visited;
    std::vector<std::pair<borders,image>> components;
    std::vector<std::pair<borders,image>> dots;


    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                auto [pixels, b] = bfs(img, p, visited);
                image im(b.height(), b.width());
                for (auto pp : pixels) {
                    im(pp.j - b.top(), pp.i - b.left()) = img(pp);
                }
                im.crop();
                if (!is_dot(im)) {
                    components.emplace_back(b, std::move(im));
                }
                else {
                    dots.emplace_back(b, std::move(im));
                }
            }
        }
    }

    std::sort(components.begin(), components.end(), [](const auto& lhs, const auto& rhs) 
            { return (lhs.first.left() + lhs.first.right()) < (rhs.first.left() + rhs.first.right()); });


    for (auto& dot : dots) {
        int dot_middle = (dot.first.left() + dot.first.right()) / 2;
        auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                return std::abs(dot_middle - lhs.first.right()) < std::abs(dot_middle - rhs.first.right());
        });

        if (component == components.end()) {
            std::cout << ".";
        }
        else {
            glue(*component, dot);
        }
    }

    int space_sum = 0;
    std::vector<int> space_diff;
    for (int i = 0; i < (int)components.size()-1; i++) {
        space_sum += std::max(components[i+1].first.left() - components[i].first.right(), 0);
        space_diff.push_back(std::max(components[i+1].first.left() - components[i].first.right(), 0));
    }

    auto get_avg_neighbors = [&](int i, int n) {
        int sum = space_diff[i];
        int current = 1;
        int taken = 0;
        bool stop = false;
        while (taken < n && !stop) {
            stop = true;

            if (i+current < space_diff.size()) {
                sum += space_diff[i+current];
                stop = false;
                ++taken;
            }

            if (i-current >= 0) {
                sum += space_diff[i-current];
                stop = false;
                ++taken;
            }

            ++current;
        }

        return (double)sum/(taken+1);
    };

    double space_avg = 0;
    if (components.size() > 1) {
        space_avg = (double)space_sum/(double)(components.size()-1);
    }

    int height_sum = 0;
    for (const auto& component : components) {
        height_sum += component.second.rows();
    }
    double height_avg = (double)height_sum/(double)components.size();

    int image_count = 0;
    std::vector<std::string> words_candidates(1,"");

    for (int i = 0; i < (int)components.size(); i++) {
        components[i].second.save("../out/" + std::to_string(line_num) + "/" + std::to_string(image_count) + ".png");
        ++image_count;

        bool not_above_height_avg = components[i].second.rows() <= height_avg;
        auto prediction = recognize(components[i].second);

        if (prediction.first == "l" && not_above_height_avg) {
            prediction.first = "c";
        }

        if (prediction.first == "j" && not_above_height_avg) {
            prediction.first = "i";
        }

        if (prediction.first == "p" && not_above_height_avg) {
            prediction.first = "e";
        }

        std::vector<std::string> words_candidates_to_add;
        for (auto& word : words_candidates) {
            if (!prediction.second.empty()) {
                words_candidates_to_add.push_back(word + prediction.second);
            }
            word += prediction.first;
        }

        for (auto& word : words_candidates_to_add) {
            words_candidates.push_back(std::move(word));
        }

        if (i != (int)components.size()-1) {
            int this_diff = std::max(components[i+1].first.left() - components[i].first.right(), 0);

            int prev_diff = (i > 0) ? std::max(components[i].first.left() - components[i-1].first.right(), 0) : 999999;
            int next_diff = (i+2 < (int)components.size()) ?  std::max(components[i+2].first.left() - components[i+1].first.right(), 0) : 99999;
            if (this_diff >= 1.6*prev_diff || this_diff >= 1.6*next_diff) {
                if (this_diff >= 1.6*space_avg || this_diff >= 1.6*get_avg_neighbors(i,10)) {
                    if (words_candidates.size() == 1) {
                        std::cout <<  words_candidates[0];
                    }
                    else {
                        bool found = false;
                        for (const auto& word : words_candidates) {
                            if (dictionary.find(word) != dictionary.end() && !(word.size() == 1 && word != "a")) {
                                std::cout <<  word;
                                found = true;
                                break;
                            }
                        }

                        if (!found) {
                            std::cout << words_candidates[0];
                        }
                    }
                    // if (words_candidates.size() > 1) {
                    //     std::cout << "{";
                    // }
                    // for (int w = 0; w < words_candidates.size(); w++) {
                    //     std::cout <<  words_candidates[w];
                    //     if (w != words_candidates.size()-1) {
                    //         std::cout << ",";
                    //     }
                    // }
                    // if (words_candidates.size() > 1) {
                    //     std::cout << "}";
                    // }

                    std::cout << " ";
                    words_candidates.clear();
                    words_candidates.push_back("");
                }
            }
        }
    }

    if (!words_candidates.empty() && !words_candidates[0].empty()) {
        // if (words_candidates.size() > 1) {
        //     std::cout << "{";
        // }
        // for (int w = 0; w < words_candidates.size(); w++) {
        //     std::cout <<  words_candidates[w];
        //     if (w != words_candidates.size()-1) {
        //         std::cout << ",";
        //     }
        // }
        // if (words_candidates.size() > 1) {
        //     std::cout << "}";
        // }
        if (words_candidates.size() == 1) {
            std::cout <<  words_candidates[0];
        }
        else {
            bool found = false;
            for (const auto& word : words_candidates) {
                if (dictionary.find(word) != dictionary.end() && !(word.size() == 1 && word != "a")) {
                    std::cout <<  word;
                    found = true;
                    break;
                }
            }

            if (!found) {
                std::cout << words_candidates[0];
            }
        }
    }
}

bool bfs(const image& img, pixel p, int line_num, int line_num_components)
{
    std::unordered_set<pixel, pixel::hash> visited;
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    int black_count = 0;

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                if (std::abs(dj) == std::abs(di)) {
                    continue;
                }
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                if ((img.check_color(neighbour, Color::black) || img.check_color(neighbour, Color::white)) && visited.find(neighbour) == visited.cend()) {
                    if (img.check_color(neighbour, Color::black)) {
                        ++black_count;
                    }
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    if (black_count > 0) {
        image line(b.height(), b.width());
        for (auto pp : visited) {
            line(pp.j - b.top(), pp.i - b.left()) = img(pp);
        }
        line.crop();
        line.save("../out/line" + std::to_string(line_num) + ".png");
        if (line_num_components == -1 || line_num == line_num_components) {
            get_components(line, line_num);
            std::cout  << std::endl;
        }
        return true;
    }

    return false;
}

void bfs(const image& img, int line_num_components)
{
    int line_num = 0;
    for (int j = 1; j < img.rows(); ++j) {
        if (j == 1 || img.check_color({j-1,0}, Color::gray)) {
            if (bfs(img, {j,0}, line_num, line_num_components)) {
                ++line_num;
            }
        }
    }
}

void process_image(const fs::path& img_name, int line_num_components)
{
    std::cout << img_name << std::endl;
    try {
        image img(img_name);
        strips s(img);
        s.connect();
        auto strips_image = s.concatenate_strips();
        auto strips_contour = s.concatenate_strips_with_lines(false);
        auto strips_contour_original = s.concatenate_strips_with_lines(true);
        auto result_image = s.result();

        result_image.save(std::string("../out/") + std::string(img_name.filename()));
        bfs(result_image, line_num_components);
    } 
    catch (std::exception& ex) {
        std::cerr << ex.what() << std::endl;
    }
}

std::pair<std::string, std::string> recognize(image& component)
{
    component.resize(28,28);
    auto t = component.get_tensor(0,1);
    std::string character =  std::string(1, char(model_one.predict_class({t}) + 97));

    if (model_one_two.predict_class({t}) == 1) {
        // we have two letters maybe
        return {character, std::string(1, char(model_first.predict_class({t}) + 97)) + std::string(1, char(model_second.predict_class({t}) + 97))};
    }

    return {character, ""};
}

int main(int argc, char** argv) {
    load_dictionary();

    if (argc < 2) {
        std::cout << "./keras2cpp path_to_img [line_num_components]" << std::endl;
        return -1;
    }

    if (fs::exists("../out")) {
        fs::remove_all("../out");
    }
    fs::create_directory("../out");

    int line_num_components = (argc == 3) ? std::stoi(argv[2]) : -1;

    process_image(argv[1], line_num_components);

    return 0;
}
